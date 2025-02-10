import asyncio
import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    stt,
    transcription,
)
from livekit.plugins import deepgram, silero

load_dotenv()

logger = logging.getLogger("transcriber")


async def _forward_transcription(
    stt_stream: stt.SpeechStream, stt_forwarder: transcription.STTSegmentsForwarder
):
    """Forward the transcription to the client and log the transcript in the console"""
    async for ev in stt_stream:
        # print(f"got event {ev}")
        if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            # you may not want to log interim transcripts, they are not final and may be incorrect
            pass
        elif ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            print(" -> ", ev.alternatives[0].text)
        elif ev.type == stt.SpeechEventType.RECOGNITION_USAGE:
            logger.debug(f"metrics: {ev.recognition_usage}")

        stt_forwarder.update(ev)


def setup_participant_entrypoints(ctx: JobContext):
    async def participant_task_1(ctx: JobContext, p: rtc.RemoteParticipant):
        # You can filter out participants you are not interested in
        # if p.identity != "some_identity_of_interest":
        # return

        logger.info(f"participant task 1 starting for {p.identity}")
        # Do something with p.attributes, p.identity, p.metadata, etc.
        # my_stuff = await fetch_stuff_from_my_db(p)

        # Do something
        await asyncio.sleep(60)
        logger.info(f"participant task done for {p.identity}")

    async def participant_task_2(ctx: JobContext, p: rtc.RemoteParticipant):
        # multiple tasks can be run concurrently for each participant
        logger.info(f"participant task 2 starting for {p.identity}")
        await asyncio.sleep(10)

    # Add participant entrypoints before calling ctx.connect
    ctx.add_participant_entrypoint(entrypoint_fnc=participant_task_1)
    ctx.add_participant_entrypoint(entrypoint_fnc=participant_task_2)


async def entrypoint(ctx: JobContext):
    logger.info(f"starting transcriber (speech to text) example, room: {ctx.room.name}")

    setup_participant_entrypoints(ctx)
    # this example uses OpenAI Whisper, but you can use assemblyai, deepgram, google, azure, etc.
    stt_impl = deepgram.STT()

    if not stt_impl.capabilities.streaming:
        assert False, "STT does not support streaming"
        # wrap with a stream adapter to use streaming semantics
        stt_impl = stt.StreamAdapter(
            stt=stt_impl,
            vad=silero.VAD.load(
                min_silence_duration=0.2,
            ),
        )

    async def transcribe_track(participant: rtc.RemoteParticipant, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        stt_forwarder = transcription.STTSegmentsForwarder(
            room=ctx.room, participant=participant, track=track
        )

        stt_stream = stt_impl.stream()
        asyncio.create_task(_forward_transcription(stt_stream, stt_forwarder))

        async for ev in audio_stream:
            stt_stream.push_frame(ev.frame)

        stt_stream.end_input()

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        # spin up a task to transcribe each track
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(transcribe_track(participant, track))

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
