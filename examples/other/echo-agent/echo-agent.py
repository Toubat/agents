import asyncio
import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.vad import VADEventType
from livekit.plugins import silero

load_dotenv()
logger = logging.getLogger("echo-agent")


# An example agent that echos each utterance from the user back to them
# the example uses a queue to buffer incoming streams, and uses VAD to detect
# when the user is done speaking.
async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")

    track_fut = asyncio.Future[rtc.Track]()

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        _publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info(f"track_subscribed: {track.kind} {participant.identity} {track}")
        if (
            track.kind == rtc.TrackKind.KIND_AUDIO
            and remote_participant.identity == participant.identity
        ):
            logger.info(participant.kind)
            track_fut.set_result(track)

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # wait for the first participant to connect
    remote_participant: rtc.Participant = await ctx.wait_for_participant()
    logger.info(f"remote_participant: {remote_participant.identity}")

    vad_stream = silero.VAD.load(
        min_speech_duration=0.2, min_silence_duration=0.6
    ).stream()

    source = rtc.AudioSource(sample_rate=48000, num_channels=1)
    track = rtc.LocalAudioTrack.create_audio_track("echo", source)
    await ctx.room.local_participant.publish_track(
        track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )

    # speech queue holds AudioFrames
    queue = asyncio.Queue(maxsize=500)  # 10 seconds of audio (1000 frames * 10ms)
    is_speaking = False
    is_echoing = False

    async def _set_state(state):
        await ctx.room.local_participant.set_attributes({"lk.agent.state": state})

    await _set_state("listening")

    async def process_remote_audio_stream():
        audio_track = await track_fut
        audio_stream = rtc.AudioStream.from_track(track=audio_track)

        async for audio_event in audio_stream:
            if is_echoing:
                continue

            vad_stream.push_frame(audio_event.frame)
            try:
                queue.put_nowait(audio_event.frame)
            except asyncio.QueueFull:
                queue.get_nowait()
                queue.put_nowait(audio_event.frame)

    async def process_vad():
        nonlocal is_speaking, is_echoing
        async for vad_event in vad_stream:
            if vad_event.type == VADEventType.INFERENCE_DONE:
                continue

            logger.info(vad_event)
            if is_echoing:  # Skip VAD processing while echoing
                continue
            if vad_event.type == VADEventType.START_OF_SPEECH:
                is_speaking = True
                frames_to_keep = 50
                frames = []
                while not queue.empty():
                    frames.append(queue.get_nowait())
                for frame in frames[-frames_to_keep:]:
                    queue.put_nowait(frame)
            elif vad_event.type == VADEventType.END_OF_SPEECH:
                is_speaking = False
                is_echoing = True
                logger.info("end of speech, playing back")
                await _set_state("speaking")
                try:
                    while not queue.empty():
                        frame = queue.get_nowait()
                        await source.capture_frame(frame)
                except asyncio.QueueEmpty:
                    pass
                finally:
                    is_echoing = False  # Reset echoing flag after playback
                    await _set_state("listening")

    await asyncio.gather(
        process_remote_audio_stream(),
        process_vad(),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        ),
    )
