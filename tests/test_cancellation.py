import queue
import threading
import unittest
from unittest.mock import Mock

from litert_proxy.engine.worker import PersistentConversationWorker
from litert_proxy.engine.generation import generation_events
from litert_proxy.models import (
    ChatCompletionRequest,
    GenerationState,
    InferenceJob,
)


class CancellationTest(unittest.TestCase):
    def test_worker_status_reports_latest_context_usage(self):
        worker = PersistentConversationWorker()
        worker._record_context_usage({
            "prompt_tokens": 1200,
            "generated_tokens": 300,
            "reasoning_tokens": 80,
            "total_tokens": 1500,
        })

        status = worker.status()

        self.assertEqual(status["context_tokens"], 1500)
        self.assertEqual(status["context_prompt_tokens"], 1200)
        self.assertEqual(status["context_generated_tokens"], 300)
        self.assertEqual(status["context_reasoning_tokens"], 80)
        self.assertTrue(status["context_estimated"])

        worker._record_context_usage({
            "total_tokens": 1500,
            "context_tokens": 1750,
        })
        self.assertEqual(worker.status()["context_tokens"], 1750)

    def test_cancel_current_does_not_interrupt_an_idle_cached_conversation(self):
        worker = PersistentConversationWorker()
        conversation = Mock()
        worker._conversation = conversation

        self.assertFalse(worker.cancel_current())
        conversation.cancel_process.assert_not_called()

    def test_pre_cancelled_generation_never_starts_litert(self):
        class Conversation:
            def send_message_async(self, *_args, **_kwargs):
                self.started = True
                return iter(())

        conversation = Conversation()
        conversation.started = False
        cancel_event = threading.Event()
        cancel_event.set()
        state = GenerationState()

        events = list(
            generation_events(
                conversation,
                "hello",
                ChatCompletionRequest(
                    messages=[{"role": "user", "content": "hello"}],
                ),
                object(),
                cancel_event,
                state,
            )
        )

        self.assertEqual(events, [])
        self.assertTrue(state.cancelled)
        self.assertFalse(conversation.started)

    def test_cancel_current_marks_a_queued_job_before_worker_assignment(self):
        worker = PersistentConversationWorker()
        job = InferenceJob(
            request=ChatCompletionRequest(
                messages=[{"role": "user", "content": "hello"}],
            ),
            messages=[{"role": "user", "content": "hello"}],
            result_queue=queue.Queue(),
        )

        worker.submit(job)

        self.assertTrue(worker.cancel_current())
        self.assertTrue(job.cancel_event.is_set())
        self.assertTrue(worker.status()["busy"])


if __name__ == "__main__":
    unittest.main()
