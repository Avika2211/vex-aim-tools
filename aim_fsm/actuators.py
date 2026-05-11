import asyncio
import os
from math import pi

from gtts import gTTS
import google.cloud
from google.cloud import texttospeech

import vex

class Actuator():
    class ActuatorLocked(Exception): pass
    class ActuatorNotHeld(Exception): pass

    def __init__(self, robot, name, stop_fn = lambda : None):
        self.robot = robot
        self.name = name
        self.holder = None
        self.started = False
        self.stop_fn = stop_fn

    def __repr__(self):
        return f"<Actuator {self.name}>"

    def lock(self, node):
        if self.holder is None:
            self.holder = node
            return True
        elif self.holder is node:
            return True
        else:
            raise self.ActuatorLocked(f'{self} locked by {self.holder}')

    def unlock(self, node):
        if self.holder is node:
            self.holder = None
        else:
            raise self.ActuatorNotHeld()

    def unlock_if_held(self, node):
        "Needed if an external event shuts down a node that might have locked the actuator."
        if self.holder is node:
            self.holder = None

    def clear(self):
        self.holder = None
        self.started = False

    def status_update(self): pass

    def complete(self):
        if self.holder:
            self.holder.complete()

class DriveActuator(Actuator):
    def __init__(self, robot):
        super().__init__(robot, 'drive')

    def stop(self):
        self.robot.robot0.stop_all_movement()

    def status_update(self):
        # Bad timing can cause a just-started motion node to appear to
        # have completed because the robot isn't moving yet; we must
        # wait until robot is seen to be moving before considering
        # looking for a stopped-moving status to detect completion.
        if not self.robot.robot0.is_stopped():
            if not self.started:
                #print('drive actuator: robot started moving for', self.holder)
                self.started = True  # started moving, now wait for completion
        elif self.holder and self.started:  # robot has just stopped; signal completion
            #print('drive actuator signaling completion to', self.holder)
            self.holder.complete()
            self.holder = None
            self.started = False

    def turn(self, node, angle_rads, turn_speed=None):
        self.lock(node)
        self.started = False
        if angle_rads > 0:
            turntype = vex.TurnType.LEFT
        else:
            turntype = vex.TurnType.RIGHT
        self.robot.world_map.pause_visibility()
        #print(f'actuator turn_for({turntype}, {abs(angle_rads)*180/pi}, {turn_speed}, {vex.TurnVelocityUnits.DPS}, {False}) for {self.holder}')
        self.robot.robot0.turn_for(turntype, abs(angle_rads)*180/pi,
                                   turn_speed, vex.TurnVelocityUnits.DPS, False)

    def forward(self, node, distance_mm, drive_speed=None):
        self.lock(node)
        self.started = False
        angle_forward = 0
        self.robot.world_map.pause_visibility()
        self.robot.robot0.move_for(distance_mm, angle_forward,
                                   drive_speed, vex.DriveVelocityUnits.MMPS, False)

    def sideways(self, node, distance_mm, drive_speed=None):
        self.lock(node)
        self.started = False
        angle_leftward = -90
        self.robot.world_map.pause_visibility()
        self.robot.robot0.move_for(distance_mm, angle_leftward,
                                   drive_speed, vex.DriveVelocityUnits.MMPS, False)

    def move_for(self, node, distance_mm, angle_deg, drive_speed=None):
        self.lock(node)
        self.started = False
        self.robot.world_map.pause_visibility()
        self.robot.robot0.move_for(distance_mm, -angle_deg,
                                   drive_speed, vex.DriveVelocityUnits.MMPS, False)

    def move_at(self, node, angle_deg, drive_speed=None):
        self.lock(node)
        self.started = False
        self.robot.world_map.pause_visibility()
        self.robot.robot0.move_at(-angle_deg, drive_speed, vex.DriveVelocityUnits.MMPS)


    def move_with_vectors(self, node, xvel, yvel, rvel):
        self.lock(node)
        self.started = False
        self.robot.world_map.pause_visibility()
        self.robot.robot0.move_with_vectors(xvel, -yvel, -rvel)

    def spin_wheels(self, node, left_vel, right_vel, back_vel):
        print('*** spin_wheels is deprecated and is going away ***')
        self.lock(node)
        self.started = False
        self.robot.world_map.pause_visibility()
        self.robot.robot0.spin_wheels(left_vel, right_vel, back_vel)


class SoundActuator(Actuator):
    def __init__(self, robot):
        super().__init__(robot, 'sound')
        self.use_gcloud = True
        self.playing = False
        self.unpause_handle = None
        self.tts_client = None
        # Google text to speech setup:
        try:
            creds = getattr(google.cloud, 'api_credentials', None)
            google_env = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            # If no credentials, will look in GOOGLE_APPLICATION_CREDENTIALS environment var
            if creds or google_env:
                self.tts_client = texttospeech.TextToSpeechClient(credentials = creds)
            self.tts_voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Journey-F",
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            self.tts_audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            synthesis_input = texttospeech.SynthesisInput(text="Hello")
            response = self.tts_client.synthesize_speech(
                input = synthesis_input,
                voice = self.tts_voice,
                audio_config = self.tts_audio_config
            )
        except Exception as e:
            print("Google text to speech:", e)
            self.tts_client = None
        # Cloud text-to-speech failed; use gTTs instead
        if self.tts_client is None:
            print('No Google Cloud credentials. Reverting to alternate speech synthesizer.')
            self.use_gcloud = False

    def status_update(self):
        if self.robot.robot0.sound.is_active():
            if not self.playing:
                self.playing = True
        else:  # sound is not active
            if self.playing is True:
                self.playing = False
                try:  # might fail if speech isn't up yet
                    self.unpause_handle = self.robot.loop.call_later(2, self.robot.speech_listener.unpause)
                except:
                    pass
                self.complete()

    def say_text(self, node, text):
        if self.robot.robot0.sound.is_active():
            print ('!!! SOUND ALREADY ACTIVE !!!')
        self.lock(node)
        if self.unpause_handle:
            self.unpause_handle.cancel()
            self.unpause_handle = None
        self.robot.loop.call_soon_threadsafe(self.launch_text_to_mp3, text)

    def launch_text_to_mp3(self, text):
        self.robot.loop.create_task(self.text_to_mp3(text))

    async def text_to_mp3(self, text):
        temp_dir = os.getenv('TEMP', '/tmp')
        speech_file_path = os.path.join(temp_dir, 'vex_speech.mp3')
        while True:
            if self.use_gcloud:
                synthesis_input = texttospeech.SynthesisInput(text=text)
                response = self.tts_client.synthesize_speech(
                    input = synthesis_input,
                    voice = self.tts_voice,
                    audio_config = self.tts_audio_config
                )
                with open(speech_file_path, 'wb') as out:
                    out.write(response.audio_content)
            else:
                tts = gTTS(text=text, lang='en')
                tts.save(speech_file_path)
            self.robot.speech_listener.pause()
            try:
                self.robot.robot0.sound.play_local_file(speech_file_path, self.robot.sound_volume)
            except vex.aim.InvalidSoundFileException:   # file too long
                print("*** Speech too long. Truncating...")
                text = text[0:len(text)//2]
                continue
            return            

    def play_sound(self, node, sound):
        self.lock(node)
        self.robot.robot0.sound.play(sound, self.robot.sound_volume)

    def play_sound_file(self, node, filepath):
        self.lock(node)
        self.robot.robot0.sound.play_local_file(filepath, self.robot.sound_volume)

    def play_note(self, node, pitch, duration):
        self.lock(node)
        self.robot.robot0.sound.play_note(pitch, duration, self.robot.sound_volume)


class KickActuator(Actuator):
    KICK_DURATION = 0.25 # seconds

    def __init__(self, robot):
        super().__init__(robot, 'kick')

    def kick(self, node, kicktype):
        self.lock(node)
        self.robot.robot0.kicker.kick(kicktype)
        self.robot.loop.call_soon_threadsafe(self.set_delayed_completion)

    def place(self, node):
        self.lock(node)
        self.robot.robot0.kicker.place()
        self.robot.loop.call_soon_threadsafe(self.set_delayed_completion)

    def set_delayed_completion(self):
        self.robot.loop.create_task(self.delayed_completion())

    async def delayed_completion(self):
        await asyncio.sleep(self.KICK_DURATION)
        if self.holder:
            self.holder.complete()


class LEDsActuator(Actuator):
    def __init__(self, robot):
        super().__init__(robot, 'leds')
        self.NUM_LEDS = 6

    def stop(self):
        self.robot.robot0.led.on(vex.LightType.ALL_LEDS, vex.Color.TRANSPARENT)

    def set_light_color(self, node, *args):
        if len(args) == 2 or len(args) == 4:
            corrected_args = args
        else:
            corrected_args = [vex.LightType.ALL_LEDS, *args]
        self.lock(node)
        self.robot.robot0.led.on(*corrected_args)


class DisplayActuator(Actuator):
    EMOJI_NAMES =  [key for (key,value) in vars(vex.EmojiType).items()
                    if isinstance(value, vex.EmojiType)]

    EMOJI_VALUES = [v for v in vars(vex.EmojiType).values()
                    if isinstance(v, vex.EmojiType)]

    def __init__(self, robot):
        super().__init__(robot, 'display')

    def show_emoji(self, node, emoji, direction=vex.EmojiLookType.LOOK_FORWARD):
        self.lock(node)
        self.robot.robot0.screen.show_emoji(emoji, direction)
        self.current_emoji = emoji

    def hide_emoji(self, node):
        self.lock(node)
        self.robot.robot0.screen.hide_emoji()
        self.current_emoji = None
