from __future__ import annotations

import argparse
import json
import math
import os
import queue
import select
import socket
import struct
import threading
import time as monotonic_time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sounddevice as sd
from ursina import (
    AmbientLight,
    Button,
    DirectionalLight,
    Entity,
    InputField,
    Sky,
    Text,
    Ursina,
    Vec3,
    application,
    camera,
    color,
    destroy,
    held_keys,
    mouse,
    scene,
    time,
    window,
)
from ursina.shaders.screenspace_shaders.camera_vertical_blur import (
    camera_vertical_blur_shader,
)
from ursina.shaders.unlit_shader import unlit_shader


VOICE_PREFIX = b"\x01"
VOICE_RATE = 24_000
VOICE_FRAMES = 480
VOICE_RANGE = 22.0
MAX_PLAYERS = 24
NICKNAME_FILE = Path(os.environ.get("APPDATA", str(Path.home()))) / "Lumen Social" / "nickname.txt"


def load_nickname(default: str) -> str:
    try:
        saved_name = NICKNAME_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return default
    except OSError as error:
        print(f"Could not load saved nickname from {NICKNAME_FILE}: {error}")
        return default
    return saved_name[:20] or default


def save_nickname(name: str) -> None:
    NICKNAME_FILE.parent.mkdir(parents=True, exist_ok=True)
    NICKNAME_FILE.write_text(name, encoding="utf-8")


class RoomServer:
    """Small UDP room server for player state and LAN voice packets."""

    def __init__(self, host: str, port: int) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.socket.setblocking(False)
        self.running = threading.Event()
        self.running.set()
        self.players: dict[tuple[str, int], dict[str, Any]] = {}
        self.next_player_id = 1
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _send_json(self, address: tuple[str, int], payload: dict[str, Any]) -> None:
        self.socket.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), address)

    def _serve(self) -> None:
        next_snapshot = 0.0
        while self.running.is_set():
            try:
                readable, _, _ = select.select([self.socket], [], [], 0.05)
            except (OSError, ValueError):
                if self.running.is_set():
                    raise
                break
            if readable:
                try:
                    packet, address = self.socket.recvfrom(65_507)
                except OSError:
                    if self.running.is_set():
                        raise
                    break

                if packet.startswith(VOICE_PREFIX):
                    player = self.players.get(address)
                    if player is not None and not player["muted"]:
                        sender_id = struct.pack("!I", player["id"])
                        outgoing = VOICE_PREFIX + sender_id + packet[len(VOICE_PREFIX):]
                        for recipient, other in tuple(self.players.items()):
                            if recipient != address and monotonic_time.monotonic() - other["last_seen"] < 8:
                                self.socket.sendto(outgoing, recipient)
                    continue

                try:
                    message = json.loads(packet.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    print(f"Ignoring malformed room message: {error}")
                    continue
                if not isinstance(message, dict):
                    continue

                message_type = message.get("type")
                if message_type == "hello":
                    player = self.players.get(address)
                    if player is None:
                        if len(self.players) >= MAX_PLAYERS:
                            self._send_json(address, {"type": "error", "message": "This room is full."})
                            continue
                        player = {
                            "id": self.next_player_id,
                            "name": str(message.get("name", "Guest"))[:20],
                            "position": [0.0, 0.0, 0.0],
                            "rotation": 0.0,
                            "muted": False,
                            "speaking": False,
                            "moving": False,
                            "last_seen": monotonic_time.monotonic(),
                        }
                        self.next_player_id += 1
                        self.players[address] = player
                    player["last_seen"] = monotonic_time.monotonic()
                    self._send_json(address, {"type": "welcome", "id": player["id"]})
                elif message_type == "state":
                    player = self.players.get(address)
                    if player is None:
                        continue
                    position = message.get("position")
                    rotation = message.get("rotation", 0.0)
                    if (
                        isinstance(position, list)
                        and len(position) == 3
                        and all(isinstance(value, (int, float)) for value in position)
                        and all(math.isfinite(value) for value in position)
                        and isinstance(rotation, (int, float))
                        and math.isfinite(rotation)
                    ):
                        player["position"] = [float(value) for value in position]
                        player["rotation"] = float(rotation)
                        player["muted"] = bool(message.get("muted", False))
                        player["speaking"] = bool(message.get("speaking", False))
                        player["moving"] = bool(message.get("moving", False))
                        player["last_seen"] = monotonic_time.monotonic()

            now = monotonic_time.monotonic()
            if now >= next_snapshot:
                expired = [
                    address
                    for address, player in self.players.items()
                    if now - player["last_seen"] > 8
                ]
                for address in expired:
                    del self.players[address]
                if self.players:
                    snapshot = {
                        "type": "snapshot",
                        "players": [
                            {
                                "id": player["id"],
                                "name": player["name"],
                                "position": player["position"],
                                "rotation": player["rotation"],
                                "muted": player["muted"],
                                "speaking": player["speaking"],
                                "moving": player["moving"],
                            }
                            for player in self.players.values()
                        ],
                    }
                    encoded = json.dumps(snapshot, separators=(",", ":")).encode("utf-8")
                    for address in tuple(self.players):
                        self.socket.sendto(encoded, address)
                next_snapshot = now + 0.1

    def close(self) -> None:
        self.running.clear()
        self.socket.close()
        self.thread.join(timeout=1)


class RoomClient:
    def __init__(self, host: str, port: int, name: str) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.connect((host, port))
        self.socket.settimeout(0.5)
        self.name = name[:20]
        self.player_id: int | None = None
        self.players: dict[int, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.voice_packets: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=8)
        self.outgoing_voice: queue.Queue[bytes] = queue.Queue(maxsize=8)
        self.error: str | None = None
        self.running = threading.Event()
        self.running.set()
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()
        self.voice_sender = threading.Thread(target=self._send_voice_packets, daemon=True)
        self.voice_sender.start()
        self._send_json({"type": "hello", "name": self.name})

    def _send_json(self, payload: dict[str, Any]) -> None:
        self.socket.send(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    def _send_voice_packets(self) -> None:
        while self.running.is_set():
            try:
                audio = self.outgoing_voice.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.socket.send(VOICE_PREFIX + audio)
            except OSError as error:
                if self.running.is_set():
                    self.error = f"Voice send failed: {error}"
                return

    def _receive(self) -> None:
        last_hello = monotonic_time.monotonic()
        while self.running.is_set():
            try:
                packet = self.socket.recv(65_507)
            except socket.timeout:
                if self.player_id is None and monotonic_time.monotonic() - last_hello >= 1:
                    try:
                        self._send_json({"type": "hello", "name": self.name})
                    except OSError as error:
                        self.error = f"Room connection failed: {error}"
                        return
                    last_hello = monotonic_time.monotonic()
                continue
            except OSError as error:
                if self.running.is_set():
                    self.error = f"Room connection failed: {error}"
                return

            if packet.startswith(VOICE_PREFIX):
                if len(packet) < 5:
                    continue
                sender_id = struct.unpack("!I", packet[1:5])[0]
                try:
                    self.voice_packets.put_nowait((sender_id, packet[5:]))
                except queue.Full:
                    try:
                        self.voice_packets.get_nowait()
                        self.voice_packets.put_nowait((sender_id, packet[5:]))
                    except queue.Empty:
                        pass
                continue

            try:
                message = json.loads(packet.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                self.error = f"Invalid data from room: {error}"
                continue
            if not isinstance(message, dict):
                continue
            if message.get("type") == "welcome":
                self.player_id = int(message["id"])
            elif message.get("type") == "error":
                self.error = str(message.get("message", "Room rejected the connection."))
            elif message.get("type") == "snapshot":
                updated: dict[int, dict[str, Any]] = {}
                for player in message.get("players", []):
                    if not isinstance(player, dict):
                        continue
                    try:
                        player_id = int(player["id"])
                        position = player["position"]
                        if len(position) != 3:
                            continue
                        updated[player_id] = {
                            **player,
                            "position": tuple(float(value) for value in position),
                        }
                    except (KeyError, TypeError, ValueError):
                        continue
                with self.lock:
                    self.players = updated

    def send_state(
        self,
        position: Vec3,
        rotation: float,
        muted: bool,
        speaking: bool,
        moving: bool,
    ) -> None:
        self._send_json(
            {
                "type": "state",
                "position": [float(position.x), float(position.y), float(position.z)],
                "rotation": float(rotation),
                "muted": muted,
                "speaking": speaking,
                "moving": moving,
            }
        )

    def send_voice(self, audio: bytes) -> None:
        try:
            self.outgoing_voice.put_nowait(audio)
        except queue.Full:
            try:
                self.outgoing_voice.get_nowait()
            except queue.Empty:
                pass
            try:
                self.outgoing_voice.put_nowait(audio)
            except queue.Full:
                pass

    def get_players(self) -> dict[int, dict[str, Any]]:
        with self.lock:
            return dict(self.players)

    def close(self) -> None:
        self.running.clear()
        self.socket.close()
        self.thread.join(timeout=1)
        self.voice_sender.join(timeout=1)


class VoiceChat:
    def __init__(self, client: RoomClient) -> None:
        self.client = client
        self.muted = False
        self.deafened = False
        self.speaking = False
        self.active = False
        self.error: str | None = None
        self.input_stream: sd.InputStream | None = None
        self.output_stream: sd.OutputStream | None = None
        try:
            self.input_stream = sd.InputStream(
                samplerate=VOICE_RATE,
                channels=1,
                dtype="float32",
                blocksize=VOICE_FRAMES,
                latency="low",
                callback=self._capture,
            )
            self.output_stream = sd.OutputStream(
                samplerate=VOICE_RATE,
                channels=2,
                dtype="float32",
                blocksize=VOICE_FRAMES,
                latency="low",
                callback=self._play,
            )
            self.input_stream.start()
            self.output_stream.start()
            self.active = True
        except sd.PortAudioError as error:
            self.error = f"Voice audio unavailable: {error}"
            print(self.error)
            self.close()

    def _capture(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info
        if status:
            self.error = f"Microphone audio warning: {status}"
        if self.muted:
            self.speaking = False
            return
        samples = indata[:, 0]
        level = float(np.sqrt(np.mean(samples * samples)))
        self.speaking = level > 0.012
        if self.speaking:
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            try:
                self.client.send_voice(pcm.tobytes())
            except OSError as error:
                self.error = f"Voice send failed: {error}"

    def _play(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        del time_info
        outdata.fill(0)
        if status:
            self.error = f"Speaker audio warning: {status}"
        if self.deafened:
            while True:
                try:
                    self.client.voice_packets.get_nowait()
                except queue.Empty:
                    break
            return

        local_position = getattr(self, "local_position", (0.0, 0.0, 0.0))
        local_yaw = getattr(self, "local_yaw", 0.0)
        players = self.client.get_players()
        latest_packets: dict[int, bytes] = {}
        while True:
            try:
                sender_id, payload = self.client.voice_packets.get_nowait()
            except queue.Empty:
                break
            latest_packets[sender_id] = payload

        for sender_id, payload in latest_packets.items():
            remote = players.get(sender_id)
            if remote is None:
                continue
            pcm = np.frombuffer(payload, dtype=np.int16)
            if pcm.size != frames:
                continue
            samples = pcm.astype(np.float32) / 32768.0
            position = remote["position"]
            dx = position[0] - local_position[0]
            dz = position[2] - local_position[2]
            distance = math.sqrt(dx * dx + dz * dz)
            if distance >= VOICE_RANGE:
                continue
            volume = max(0.0, 1.0 - distance / VOICE_RANGE)
            yaw = math.radians(local_yaw)
            right = dx * math.cos(yaw) - dz * math.sin(yaw)
            pan = max(-1.0, min(1.0, right / max(distance, 0.001)))
            left_gain = math.sqrt((1.0 - pan) * 0.5) * volume
            right_gain = math.sqrt((1.0 + pan) * 0.5) * volume
            outdata[:, 0] += samples * left_gain
            outdata[:, 1] += samples * right_gain
        np.clip(outdata, -1.0, 1.0, out=outdata)

    def close(self) -> None:
        if self.input_stream is not None:
            self.input_stream.close()
            self.input_stream = None
        if self.output_stream is not None:
            self.output_stream.close()
            self.output_stream = None
        self.active = False


class RoomLobby(Entity):
    def __init__(
        self,
        name: str,
        address: str,
        port: int,
        on_start: Callable[[str, str, bool], None],
    ) -> None:
        super().__init__(parent=camera.ui, name="room_lobby")
        self.on_start = on_start
        self.port = port
        window.color = color.rgb32(7, 11, 26)
        window.title = "Voice-chat-for-everyone"
        window.size = (1440, 900)
        camera.ui_lens.set_film_size(
            camera.ui.scale_y * window.aspect_ratio,
            camera.ui.scale_y,
        )
        mouse.locked = False
        mouse.visible = True

        Entity(
            parent=self,
            model="quad",
            position=(0, 0, 0.5),
            scale=(camera.aspect_ratio * 2, 2),
            color=color.rgb32(8, 12, 29),
            shader=unlit_shader,
            double_sided=True,
        )
        Entity(
            parent=self,
            model="quad",
            position=(0, 0, 0.4),
            scale=(0.76, 0.82),
            color=color.rgb32(23, 34, 68),
            shader=unlit_shader,
            double_sided=True,
        )
        Text(
            parent=self,
            text="L U M E N",
            position=(-0.28, 0.29),
            scale=2.2,
            color=color.rgb32(92, 220, 255),
        )
        Text(
            parent=self,
            text="SOCIAL PLAZA",
            position=(-0.28, 0.22),
            scale=1.05,
            color=color.rgb32(221, 185, 255),
        )
        Text(
            parent=self,
            text="Choose a display name and join a voice room.",
            position=(-0.28, 0.14),
            scale=0.8,
            color=color.rgb32(193, 207, 237),
        )
        Text(
            parent=self,
            text="USERNAME",
            position=(-0.28, 0.075),
            scale=0.62,
            color=color.rgb32(129, 179, 224),
        )
        self.name_field = InputField(
            parent=self,
            position=(0, 0.015),
            scale=(0.56, 0.06),
            default_value=name,
            character_limit=20,
            color=color.rgb32(31, 44, 78),
            text_color=color.rgb32(245, 248, 255),
        )
        Text(
            parent=self,
            text="ROOM ADDRESS (JOIN)",
            position=(-0.28, -0.075),
            scale=0.62,
            color=color.rgb32(129, 179, 224),
        )
        self.address_field = InputField(
            parent=self,
            position=(0, -0.135),
            scale=(0.56, 0.06),
            default_value=address,
            character_limit=64,
            color=color.rgb32(31, 44, 78),
            text_color=color.rgb32(245, 248, 255),
        )
        Button(
            parent=self,
            text="HOST ROOM",
            position=(-0.15, -0.245),
            scale=(0.25, 0.075),
            color=color.rgb32(39, 157, 173),
            text_color=color.white,
            on_click=self.host_room,
        )
        Button(
            parent=self,
            text="JOIN ROOM",
            position=(0.15, -0.245),
            scale=(0.25, 0.075),
            color=color.rgb32(125, 77, 191),
            text_color=color.white,
            on_click=self.join_room,
        )
        self.status_text = Text(
            parent=self,
            text=f"Rooms use UDP port {port}. Share your LAN IP with friends.",
            position=(-0.28, -0.34),
            scale=0.62,
            color=color.rgb32(174, 190, 222),
        )
        Text(
            parent=self,
            text="WASD to move after joining  |  M to mute  |  N to deafen",
            position=(-0.28, -0.405),
            scale=0.58,
            color=color.rgb32(133, 151, 188),
        )

    def host_room(self) -> None:
        self._start(host_mode=True)

    def join_room(self) -> None:
        self._start(host_mode=False)

    def _start(self, host_mode: bool) -> None:
        name = self.name_field.text.strip()[:20] or "Guest"
        address = self.address_field.text.strip()
        if not host_mode and not address:
            self.show_error("Enter the host's IP address to join.")
            return
        try:
            save_nickname(name)
        except OSError as error:
            print(f"Could not save nickname to {NICKNAME_FILE}: {error}")
        self.on_start(name, address, host_mode)

    def show_error(self, message: str) -> None:
        self.status_text.text = message
        self.status_text.color = color.rgb32(255, 135, 145)


class PauseMenu(Entity):
    def __init__(self, on_resume: Callable[[], None], on_return: Callable[[], None]) -> None:
        super().__init__(parent=camera.ui, name="pause_menu", ignore_paused=True)
        self.on_resume = on_resume
        self.on_return = on_return
        Entity(
            parent=self,
            model="quad",
            position=(0, 0, 0.5),
            scale=(camera.aspect_ratio * 2, 2),
            color=color.rgba(5 / 255, 8 / 255, 20 / 255, 0.72),
            shader=unlit_shader,
            double_sided=True,
            ignore_paused=True,
        )
        Entity(
            parent=self,
            model="quad",
            position=(0, 0, 0.4),
            scale=(0.64, 0.72),
            color=color.rgb32(23, 34, 68),
            shader=unlit_shader,
            double_sided=True,
            ignore_paused=True,
        )
        Text(
            parent=self,
            text="PAUSED",
            position=(-0.12, 0.23),
            scale=1.8,
            color=color.rgb32(92, 220, 255),
            ignore_paused=True,
        )
        Button(
            parent=self,
            text="RESUME TO GAME",
            position=(0, 0.08),
            scale=(0.42, 0.075),
            color=color.rgb32(39, 157, 173),
            text_color=color.white,
            on_click=self.on_resume,
            ignore_paused=True,
        )
        Button(
            parent=self,
            text="RETURN TO LOBBY",
            position=(0, -0.025),
            scale=(0.42, 0.075),
            color=color.rgb32(125, 77, 191),
            text_color=color.white,
            on_click=self.on_return,
            ignore_paused=True,
        )
        Button(
            parent=self,
            text="QUIT",
            position=(0, -0.13),
            scale=(0.42, 0.075),
            color=color.rgb32(166, 67, 89),
            text_color=color.white,
            on_click=application.quit,
            ignore_paused=True,
        )


class SocialWorld(Entity):
    def __init__(
        self,
        name: str,
        client: RoomClient,
        host_mode: bool,
        server_address: str,
        on_return_to_lobby: Callable[[], None],
    ) -> None:
        self._preexisting_scene_entities = set(scene.entities)
        super().__init__(name="social_world")
        self.ignore_paused = True
        self.name = name
        self.client = client
        self.host_mode = host_mode
        self.server_address = server_address
        self.on_return_to_lobby = on_return_to_lobby
        self.pause_menu: PauseMenu | None = None
        self.voice = VoiceChat(client)
        self.player = Entity(position=(0, 0, 0), rotation_y=0)
        self.look_pivot = Entity(parent=self.player, position=(0, 1.65, 0))
        self.velocity_y = 0.0
        self.grounded = True
        self.crouching = False
        self.elapsed = 0.0
        self.remote_avatars: dict[int, dict[str, Any]] = {}
        self.solid_obstacles: list[tuple[float, float, float, float]] = []
        self._create_window()
        self._create_world()
        self._create_hud()
        camera.parent = self.look_pivot
        camera.position = (0, 0, 0)
        camera.rotation = (0, 0, 0)
        camera.fov = 92
        mouse.locked = False
        mouse.visible = True
        self.voice.local_position = tuple(float(value) for value in self.player.position)
        self.voice.local_yaw = 0.0

    @staticmethod
    def _create_window() -> None:
        window.title = "Voice-chat-for-everyone"
        window.color = color.rgb32(7, 11, 26)
        window.size = (1440, 900)
        window.exit_button.visible = False
        window.fps_counter.enabled = False
        sky = Sky()
        sky.color = color.rgb32(10, 16, 36)
        scene.ambient_color = color.rgb32(35, 43, 76)
        AmbientLight(color=color.rgba(80 / 255, 95 / 255, 150 / 255, 0.35))
        sun = DirectionalLight(
            color=color.rgba(150 / 255, 185 / 255, 255 / 255, 0.65)
        )
        sun.look_at(Vec3(1, -2, -1))

    def _create_world(self) -> None:
        Entity(
            model="plane",
            scale=(180, 1, 180),
            texture="white_cube",
            texture_scale=(120, 120),
            color=color.rgb32(18, 26, 48),
            collider="box",
        )
        Entity(
            model="cube",
            scale=(27, 0.36, 27),
            position=(0, 0.17, 0),
            color=color.rgb32(18, 29, 58),
        )
        Entity(
            model="cube",
            scale=(22, 0.08, 22),
            position=(0, 0.39, 0),
            color=color.rgb32(34, 48, 82),
        )
        for radius, tint in (
            (11.2, color.rgb32(75, 220, 255)),
            (13.2, color.rgb32(161, 93, 255)),
        ):
            for index in range(24):
                angle = math.tau * index / 24
                Entity(
                    model="cube",
                    position=(math.sin(angle) * radius, 0.45, math.cos(angle) * radius),
                    scale=(0.55, 0.045, 0.55),
                    color=tint,
                )

        for index in range(16):
            angle = math.tau * index / 16
            x, z = math.sin(angle) * 42, math.cos(angle) * 42
            height = 9 + (index * 7 % 13)
            tint = color.rgb32(
                24 + index % 4 * 8,
                43 + index % 3 * 8,
                81 + index % 5 * 7,
            )
            Entity(
                model="cube",
                position=(x, height / 2, z),
                scale=(7, height, 7),
                color=tint,
                collider="box",
            )
            self.solid_obstacles.append((x, z, 3.7, 3.7))
            for floor in range(2, int(height), 4):
                Entity(
                    model="cube",
                    position=(x, floor, z - 3.56),
                    scale=(5.9, 0.08, 0.08),
                    color=color.rgb32(51, 195, 255)
                    if index % 2
                    else color.rgb32(214, 103, 255),
                )

        for index in range(8):
            angle = math.tau * index / 8
            x, z = math.sin(angle) * 18, math.cos(angle) * 18
            Entity(
                model="cube",
                position=(x, 2.7, z),
                scale=(0.32, 5.2, 0.32),
                color=color.rgb32(77, 108, 165),
            )
            Entity(
                model="sphere",
                position=(x, 5.5, z),
                scale=0.62,
                color=color.rgb32(92, 225, 255)
                if index % 2
                else color.rgb32(218, 112, 255),
            )

        for index in range(12):
            angle = math.tau * index / 12
            x, z = math.sin(angle) * 31, math.cos(angle) * 31
            tree_height = 4.2 + (index % 3) * 0.35
            Entity(
                model="cube",
                position=(x, tree_height / 2, z),
                scale=(0.48, tree_height, 0.48),
                rotation_z=(index % 3 - 1) * 3,
                color=color.rgb32(112, 70, 43),
                collider="box",
            )
            self.solid_obstacles.append((x, z, 0.55, 0.55))
            branch_color = color.rgb32(112, 70, 43)
            for branch_index, side in enumerate((-1, 1)):
                branch = Entity(
                    model="cube",
                    position=(x + side * 0.55, 2.1 + branch_index * 0.35, z),
                    scale=(0.22, 1.45, 0.22),
                    rotation_z=side * -38,
                    color=branch_color,
                )
                branch.rotation_y = index * 30
            leaf_colors = (
                color.rgb32(35, 133, 96),
                color.rgb32(49, 157, 105),
                color.rgb32(75, 177, 117),
            )
            for cluster_index, (dx, dy, dz, size) in enumerate(
                (
                    (0, tree_height + 0.3, 0, 2.0),
                    (-0.8, tree_height - 0.1, 0.15, 1.55),
                    (0.8, tree_height - 0.05, -0.1, 1.6),
                    (0.1, tree_height + 1.0, 0.45, 1.55),
                    (-0.2, tree_height + 0.45, -0.85, 1.5),
                )
            ):
                Entity(
                    model="sphere",
                    position=(x + dx, dy, z + dz),
                    scale=(size, size * 0.86, size),
                    color=leaf_colors[(cluster_index + index) % len(leaf_colors)],
                )

        for index in range(8):
            angle = math.tau * index / 8
            x, z = math.sin(angle) * 7.5, math.cos(angle) * 7.5
            half_width = abs(math.cos(angle)) * 1.05 + abs(math.sin(angle)) * 0.35
            half_depth = abs(math.sin(angle)) * 1.05 + abs(math.cos(angle)) * 0.35
            Entity(
                model="cube",
                position=(x, 0.85, z),
                scale=(2.1, 1.4, 0.7),
                rotation_y=-math.degrees(angle),
                color=color.rgb32(55, 70, 108),
                collider="box",
            )
            self.solid_obstacles.append((x, z, half_width, half_depth))

    def _create_hud(self) -> None:
        left_edge = window.left.x + 0.04
        Text(
            text="L U M E N   /   SOCIAL PLAZA",
            position=(left_edge, 0.455),
            scale=min(1.25, max(0.55, window.aspect_ratio * 0.85)),
            color=color.rgb32(147, 222, 255),
        )
        self.room_text = Text(
            text="",
            position=(left_edge, 0.405),
            scale=min(0.75, max(0.45, window.aspect_ratio * 0.75)),
            color=color.rgb32(184, 196, 221),
        )
        self.voice_text = Text(
            text="",
            position=(left_edge, 0.34),
            scale=min(0.8, max(0.45, window.aspect_ratio * 0.75)),
            color=color.rgb32(106, 245, 197),
        )
        self.players_text = Text(
            text="",
            position=(window.right.x - 0.04, 0.39),
            origin=(0.5, 0),
            scale=min(0.8, max(0.45, window.aspect_ratio * 0.75)),
            color=color.rgb32(226, 233, 250),
        )
        Text(
            text="WASD move   SPACE jump   SHIFT sprint   CTRL crouch\n"
            "Move mouse to look   Click to lock cursor   M mute   N deafen   ESC pause",
            position=(left_edge, -0.43),
            scale=min(0.72, max(0.4, window.aspect_ratio * 0.75)),
            color=color.rgb32(185, 198, 224),
        )
        Entity(
            parent=camera.ui,
            model="circle",
            scale=0.006,
            color=color.rgb32(197, 231, 255),
        )

    def update(self) -> None:
        dt = min(time.dt, 0.05)
        self.elapsed += dt
        moving = False
        if not application.paused:
            if mouse.velocity.length() > 0:
                self.player.rotation_y += mouse.velocity[0] * 40
                self.look_pivot.rotation_x = max(
                    -85, min(85, self.look_pivot.rotation_x - mouse.velocity[1] * 40)
                )

            self.crouching = bool(held_keys["left control"] or held_keys["right control"])
            self.look_pivot.y = 1.05 if self.crouching else 1.65
            forward = Vec3(
                math.sin(math.radians(self.player.rotation_y)),
                0,
                math.cos(math.radians(self.player.rotation_y)),
            )
            right = Vec3(forward.z, 0, -forward.x)
            direction = Vec3(0, 0, 0)
            direction += forward * (held_keys["w"] - held_keys["s"])
            direction += right * (held_keys["d"] - held_keys["a"])
            previous_position = Vec3(self.player.position)
            if direction.length() > 0:
                direction = direction.normalized()
                speed = 3.0 if self.crouching else (9.5 if held_keys["left shift"] else 6.0)
                next_x = self.player.x + direction.x * speed * dt
                if self._can_move_to(next_x, self.player.z):
                    self.player.x = next_x
                next_z = self.player.z + direction.z * speed * dt
                if self._can_move_to(self.player.x, next_z):
                    self.player.z = next_z
            moving = (self.player.position - previous_position).length() > 0.001

            if self.grounded and held_keys["space"]:
                self.velocity_y = 7.0
                self.grounded = False
            self.velocity_y -= 19.0 * dt
            self.player.y += self.velocity_y * dt
            if self.player.y <= 0:
                self.player.y = 0
                self.velocity_y = 0
                self.grounded = True

        self.voice.local_position = tuple(float(value) for value in self.player.position)
        self.voice.local_yaw = float(self.player.rotation_y)
        if self.elapsed >= 0.05:
            self.elapsed = 0
            self.client.send_state(
                self.player.position,
                self.player.rotation_y,
                self.voice.muted,
                self.voice.speaking,
                moving,
            )
            self._update_remote_players()
            self._update_hud()

    def _can_move_to(self, x: float, z: float) -> bool:
        radius = 0.38
        for obstacle_x, obstacle_z, half_width, half_depth in self.solid_obstacles:
            nearest_x = max(obstacle_x - half_width, min(x, obstacle_x + half_width))
            nearest_z = max(obstacle_z - half_depth, min(z, obstacle_z + half_depth))
            dx = x - nearest_x
            dz = z - nearest_z
            if dx * dx + dz * dz < radius * radius:
                return False
        return True

    @staticmethod
    def _create_humanoid(parent: Entity, player_id: int) -> dict[str, Entity]:
        palette = (
            (color.rgb32(54, 176, 191), color.rgb32(230, 173, 96)),
            (color.rgb32(142, 92, 193), color.rgb32(94, 171, 205)),
            (color.rgb32(210, 105, 105), color.rgb32(102, 177, 125)),
            (color.rgb32(73, 122, 199), color.rgb32(222, 171, 120)),
            (color.rgb32(196, 132, 55), color.rgb32(129, 107, 190)),
        )
        shirt_color, pants_color = palette[player_id % len(palette)]
        skin_color = color.rgb32(232, 183, 146)
        parts: dict[str, Entity] = {}

        parts["torso"] = Entity(
            parent=parent,
            model="sphere",
            position=(0, 1.2, 0),
            scale=(0.64, 0.82, 0.38),
            color=shirt_color,
        )
        Entity(
            parent=parent,
            model="sphere",
            position=(0, 0.72, 0),
            scale=(0.48, 0.3, 0.35),
            color=pants_color,
        )
        Entity(
            parent=parent,
            model="sphere",
            position=(0, 1.68, 0),
            scale=(0.2, 0.25, 0.2),
            color=skin_color,
        )
        head = Entity(parent=parent, position=(0, 1.98, 0))
        parts["head"] = head
        Entity(
            parent=head,
            model="sphere",
            scale=(0.43, 0.48, 0.4),
            color=skin_color,
        )
        Entity(
            parent=head,
            model="sphere",
            position=(0, 0.19, -0.025),
            scale=(0.45, 0.25, 0.42),
            color=color.rgb32(50, 42, 53),
        )
        for eye_x in (-0.12, 0.12):
            Entity(
                parent=head,
                model="sphere",
                position=(eye_x, 0.04, 0.365),
                scale=(0.07, 0.09, 0.035),
                color=color.rgb32(35, 43, 57),
            )
        Entity(
            parent=head,
            model="sphere",
            position=(0, -0.045, 0.39),
            scale=(0.075, 0.11, 0.06),
            color=color.rgb32(218, 157, 128),
        )

        for side, key in ((-1, "left_arm"), (1, "right_arm")):
            shoulder = Entity(parent=parent, position=(side * 0.37, 1.47, 0))
            parts[key] = shoulder
            Entity(
                parent=shoulder,
                model="sphere",
                position=(side * 0.035, -0.23, 0),
                scale=(0.25, 0.42, 0.25),
                color=shirt_color,
            )
            Entity(
                parent=shoulder,
                model="sphere",
                position=(side * 0.04, -0.48, 0.015),
                scale=(0.2, 0.18, 0.2),
                color=skin_color,
            )

        for side, key in ((-1, "left_leg"), (1, "right_leg")):
            hip = Entity(parent=parent, position=(side * 0.17, 0.67, 0))
            parts[key] = hip
            Entity(
                parent=hip,
                model="sphere",
                position=(0, -0.22, 0),
                scale=(0.25, 0.48, 0.27),
                color=pants_color,
            )
            Entity(
                parent=hip,
                model="sphere",
                position=(0, -0.52, 0.025),
                scale=(0.2, 0.2, 0.22),
                color=color.rgb32(43, 48, 65),
            )
            Entity(
                parent=hip,
                model="sphere",
                position=(0, -0.61, 0.13),
                scale=(0.25, 0.12, 0.38),
                color=color.rgb32(31, 37, 53),
            )
        return parts

    def _update_remote_players(self) -> None:
        local_id = self.client.player_id
        for player_id, state in self.client.get_players().items():
            if player_id == local_id:
                continue
            if player_id not in self.remote_avatars:
                avatar = Entity(position=state["position"])
                parts = self._create_humanoid(avatar, player_id)
                label = Text(
                    text=state["name"],
                    parent=avatar,
                    position=(0, 2.85, 0),
                    billboard=True,
                    scale=4,
                    color=color.rgb32(226, 238, 255),
                )
                indicator = Entity(
                    parent=avatar,
                    model="sphere",
                    position=(0.42, 2.85, 0),
                    scale=0.13,
                    color=color.rgb32(255, 90, 110),
                )
                self.remote_avatars[player_id] = {
                    "root": avatar,
                    "label": label,
                    "indicator": indicator,
                    "parts": parts,
                    "walk_phase": 0.0,
                }
            avatar_data = self.remote_avatars[player_id]
            avatar = avatar_data["root"]
            target = Vec3(*state["position"])
            moving = bool(state.get("moving", False))
            avatar.position = avatar.position + (target - avatar.position) * 0.35
            avatar.rotation_y = float(state.get("rotation", 0.0))
            if moving:
                avatar_data["walk_phase"] += time.dt * 11
            swing = math.sin(avatar_data["walk_phase"]) * (28 if moving else 0)
            parts = avatar_data["parts"]
            parts["left_leg"].rotation_x = swing
            parts["right_leg"].rotation_x = -swing
            parts["left_arm"].rotation_x = -swing * 0.75
            parts["right_arm"].rotation_x = swing * 0.75
            parts["torso"].y = 1.2 + (
                abs(math.sin(avatar_data["walk_phase"] * 2)) * 0.045 if moving else 0
            )
            avatar_data["label"].text = str(state["name"])
            indicator = avatar_data["indicator"]
            indicator.color = (
                color.rgb32(95, 244, 182)
                if state.get("speaking") and not state.get("muted")
                else color.rgb32(255, 90, 110)
                if state.get("muted")
                else color.rgb32(120, 139, 173)
            )

        current_ids = set(self.client.get_players())
        for player_id in set(self.remote_avatars) - current_ids:
            avatar_data = self.remote_avatars.pop(player_id)
            destroy(avatar_data["root"])

    def _update_hud(self) -> None:
        joined = self.client.player_id is not None
        mode = "HOSTING" if self.host_mode else "CONNECTED"
        self.room_text.text = (
            f"{mode}  {self.server_address}"
            if joined
            else f"Connecting to {self.server_address}..."
        )
        if self.client.error:
            self.room_text.text = self.client.error
            self.room_text.color = color.rgb32(255, 135, 145)
        else:
            self.room_text.color = color.rgb32(184, 196, 221)

        if self.voice.error:
            voice_state = self.voice.error
            self.voice_text.color = color.rgb32(255, 175, 120)
        elif not self.voice.active:
            voice_state = "VOICE OFFLINE  |  Check microphone and speakers"
            self.voice_text.color = color.rgb32(255, 175, 120)
        else:
            mic = "MIC MUTED" if self.voice.muted else ("MIC LIVE" if self.voice.speaking else "MIC READY")
            ears = "DEAFENED" if self.voice.deafened else "AUDIO ON"
            voice_state = f"{mic}   /   {ears}"
            self.voice_text.color = (
                color.rgb32(255, 120, 140)
                if self.voice.muted or self.voice.deafened
                else color.rgb32(106, 245, 197)
            )
        self.voice_text.text = voice_state

        players = self.client.get_players()
        local_id = self.client.player_id
        names = [
            f"{player['name']}  (you)" if player_id == local_id else str(player["name"])
            for player_id, player in players.items()
        ]
        if local_id not in players:
            names.insert(0, f"{self.name}  (you)")
        self.players_text.text = (
            f"PLAYERS  {len(players)}/{MAX_PLAYERS}\n"
            + "\n".join(f"  {name}" for name in names[:8])
        )

    def input(self, key: str) -> None:
        if key == "escape":
            if application.paused:
                if self.pause_menu is not None:
                    self.pause_menu.on_resume()
            else:
                self._show_pause_menu()
        elif application.paused:
            return
        elif key == "m":
            self.voice.muted = not self.voice.muted
            if self.voice.muted:
                self.voice.speaking = False
        elif key == "n":
            self.voice.deafened = not self.voice.deafened
        elif key == "left mouse down" and not mouse.locked:
            mouse.locked = True
            mouse.visible = False

    def _show_pause_menu(self) -> None:
        application.pause()
        camera.shader = camera_vertical_blur_shader
        camera.set_shader_input("blur_size", 0.035)
        mouse.locked = False
        mouse.visible = True
        self.pause_menu = PauseMenu(self._resume, self._return_to_lobby)

    def _resume(self) -> None:
        self._destroy_pause_menu()
        self._clear_camera_shader()
        application.resume()
        mouse.locked = True
        mouse.visible = False

    def _return_to_lobby(self) -> None:
        self._destroy_pause_menu()
        self._clear_camera_shader()
        application.resume()
        mouse.locked = False
        mouse.visible = True
        camera.parent = scene
        camera.position = (0, 0, 0)
        camera.rotation = (0, 0, 0)
        camera.fov = 90
        self.voice.close()
        for entity in reversed(scene.entities[:]):
            if (
                entity not in self._preexisting_scene_entities
                and entity not in scene._entities_marked_for_removal
            ):
                destroy(entity)
        self.on_return_to_lobby()

    @staticmethod
    def _clear_camera_shader() -> None:
        filter_manager = getattr(camera, "filter_manager", None)
        if filter_manager is not None:
            filter_manager.cleanup()
            camera.filter_manager = None
        filter_quad = getattr(camera, "filter_quad", None)
        if filter_quad is not None:
            filter_quad.removeNode()
            camera.filter_quad = None
        camera._shader = None

    def _destroy_pause_menu(self) -> None:
        if self.pause_menu is None:
            return
        menu = self.pause_menu
        for entity in reversed(scene.entities[:]):
            if entity is menu or entity.has_ancestor(menu):
                destroy(entity)
        self.pause_menu = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lumen Social, a stylized 3D hangout with LAN proximity voice chat."
    )
    parser.add_argument("--name", default="Guest", help="Your display name (up to 20 characters).")
    parser.add_argument(
        "--join",
        metavar="HOST",
        help="Join a room hosted on this LAN address instead of hosting your own room.",
    )
    parser.add_argument("--port", type=int, default=8765, help="LAN room UDP port (default: 8765).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name = load_nickname(args.name.strip()[:20] or "Guest")
    game = Ursina(title="Voice-chat-for-everyone")
    server: RoomServer | None = None
    client: RoomClient | None = None
    world: SocialWorld | None = None

    def start_room(player_name: str, address: str, host_mode: bool) -> None:
        nonlocal server, client, world
        try:
            if host_mode:
                server = RoomServer("0.0.0.0", args.port)
                server_address = f"LAN room / UDP {args.port}"
                connect_host = "127.0.0.1"
                print(
                    f"Hosting Lumen Social on UDP port {args.port}. "
                    "Share this computer's LAN IP to invite friends."
                )
            else:
                server_address = f"{address}:{args.port}"
                connect_host = address
            client = RoomClient(connect_host, args.port, player_name)
        except OSError as error:
            if client is not None:
                client.close()
                client = None
            if server is not None:
                server.close()
                server = None
            lobby.show_error(f"Could not start room: {error}")
            return

        lobby.enabled = False

        def return_to_lobby() -> None:
            nonlocal server, client, world
            if client is not None:
                client.close()
                client = None
            if server is not None:
                server.close()
                server = None
            world = None
            lobby.enabled = True
            lobby.status_text.text = "You left the room. Host or join another room."
            lobby.status_text.color = color.rgb32(174, 190, 222)

        world = SocialWorld(
            player_name,
            client,
            host_mode,
            server_address,
            return_to_lobby,
        )

    lobby = RoomLobby(name, args.join or "127.0.0.1", args.port, start_room)
    try:
        if args.join:
            start_room(name, args.join, host_mode=False)
        game.run()
    finally:
        if world is not None:
            world.voice.close()
        if client is not None:
            client.close()
        if server is not None:
            server.close()


if __name__ == "__main__":
    main()