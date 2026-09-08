import unittest

from experiments.auto_prompt.experiment import (
    TutorJudgeResult,
    deterministic_violations,
    score_judgement,
)


class AutoPromptScoringTests(unittest.TestCase):
    def test_deterministic_question_limit(self):
        violations = deterministic_violations(
            "First question? Second question?",
            {"max_questions": 1},
        )
        self.assertEqual(len(violations), 1)
        self.assertIn("max_questions", violations[0])

    def test_hard_violation_zeroes_score(self):
        judgement = TutorJudgeResult(
            instruction_adherence=4,
            behavior_preservation=4,
            pedagogical_correctness=4,
            providing_guidance=4,
            answer_control=4,
            actionability=4,
            coherence=4,
            tutor_tone=4,
            human_likeness=4,
            source_teaching_action="give_hint",
            target_teaching_action="give_hint",
            hard_constraint_violations=["revealed answer too early"],
            feedback="Otherwise good.",
        )
        score, components, violations = score_judgement(judgement, [])
        self.assertEqual(score, 0.0)
        self.assertEqual(components["instruction_adherence"], 1.0)
        self.assertEqual(violations, ["revealed answer too early"])

    def test_perfect_soft_score(self):
        judgement = TutorJudgeResult(
            instruction_adherence=4,
            behavior_preservation=4,
            pedagogical_correctness=4,
            providing_guidance=4,
            answer_control=4,
            actionability=4,
            coherence=4,
            tutor_tone=4,
            human_likeness=4,
            source_teaching_action="ask_guiding_question",
            target_teaching_action="ask_guiding_question",
            hard_constraint_violations=[],
            feedback="Good.",
        )
        score, _, violations = score_judgement(judgement, [])
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
