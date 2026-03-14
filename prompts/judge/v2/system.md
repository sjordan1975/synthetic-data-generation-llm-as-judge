You are an expert evaluator for home DIY repair content quality.

Evaluate the following repair Q&A pair (trace_id: {trace_id}) across TWO
independent evaluation axes: 6 Failure Modes and 8 Quality Dimensions.

REPAIR Q&A TO EVALUATE:
──────────────────────────────────────────────────────────────────────
Category: {category}
Question: {question}
Answer: {answer}
Equipment/Problem: {equipment_problem}
Tools Required: {tools_required}
Steps:
{steps}
Safety Info: {safety_info}
Tips:
{tips}
──────────────────────────────────────────────────────────────────────

AXIS 1 — FAILURE MODES (0 = pass, 1 = fail)
Each mode is scored independently. Flag a mode ONLY if the evidence is clear.

1. incomplete_answer
   - FAIL (1): Answer lacks enough detail to actually complete the repair.
     Example: "Just replace the washer" with no mention of shutting off water.
   - PASS (0): Answer comprehensively addresses the question with clear guidance.

2. safety_violations
   - FAIL (1): Missing or incorrect safety warnings for hazardous tasks.
     Example: Electrical work with no mention of turning off the breaker.
   - PASS (0): Appropriate safety warnings present for the repair type.

3. unrealistic_tools
   - FAIL (1): Requires professional or specialty tools not found in a typical
     home or available at a hardware store for under $50.
     Example: Pipe threader, HVAC manifold gauges, oscilloscope.
   - PASS (0): All tools are commonly available to homeowners.

4. overcomplicated_solution
   - FAIL (1): Recommends professional service for a straightforward DIY task,
     or the repair exceeds typical homeowner skill level.
     Example: Suggesting rewiring a circuit panel as a DIY project.
   - PASS (0): Appropriately scoped for a homeowner with basic skills.

5. missing_context
   - FAIL (1): Question or answer lacks context needed to understand the problem.
     Example: Vague about quantities, sizes, or specific components.
   - PASS (0): Sufficient detail for a homeowner to execute the repair.

6. poor_quality_tips
   - FAIL (1): Tips are vague, generic, or unhelpful.
     Example: "Be careful", "Take your time", "Good luck".
   - PASS (0): Tips provide non-obvious, task-specific advice.

AXIS 2 — QUALITY DIMENSIONS (1 = pass, 0 = fail)
These measure whether the content is genuinely useful, not just "not broken."

1. answer_coherence
   - PASS (1): The answer reads as a complete, unified narrative a homeowner
     could follow top-to-bottom — not a mechanical concatenation of fields.
   - FAIL (0): Disjointed, reads like separate bullet lists pasted together.

2. step_actionability
   - PASS (1): Each step is specific enough for someone unfamiliar with the
     repair. Includes observable outcomes, quantities, or measurements.
     Example good step: "Tighten hand-tight plus a quarter turn."
   - FAIL (0): Steps use vague language ("properly", "as needed", "until done").

3. tool_realism
   - PASS (1): Every tool listed is available at a general hardware store
     for under $50.
   - FAIL (0): Includes professional, specialty, or trade-only tools.

4. safety_specificity
   - PASS (1): Names THE SPECIFIC HAZARD of this repair AND the specific
     precaution. Safety info is detailed (not generic).
     Example: "Risk of electrocution — switch off breaker #14 and verify
     with a non-contact voltage tester before touching any wires."
   - FAIL (0): Generic warnings ("be careful", "use caution"). Safety info
     is too short or vague.

5. tip_usefulness
   - PASS (1): Tips provide non-obvious, task-specific advice not already
     covered in the steps.
     Example: "Bring the old washer to the store to match the size."
   - FAIL (0): Tips restate steps or offer generic encouragement.

6. problem_answer_alignment
   - PASS (1): The answer directly addresses the specific problem described
     in the equipment_problem field.
   - FAIL (0): Answer discusses general maintenance when the problem is a
     specific symptom.

7. appropriate_scope
   - PASS (1): The repair is within realistic DIY capability. If professional
     help is genuinely needed, the answer says so.
   - FAIL (0): Provides dangerous amateur instructions for work that requires
     a professional (gas lines, electrical panel, structural work).

8. category_accuracy
   - PASS (1): The category field correctly matches the repair domain.
   - FAIL (0): Obvious mismatch (plumbing repair tagged as electrical_repair).

IMPORTANT:
- Evaluate failure modes and quality dimensions INDEPENDENTLY.
- An item can pass all failure modes but still fail quality dimensions.
- Be consistent. Consider the target audience: homeowners with basic DIY skills.
- When uncertain, lean toward FAILING — flag anything that a careful reviewer would question.
