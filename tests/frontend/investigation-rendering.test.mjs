import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  hypothesisActionLabel,
  normalizeInvestigationResponse,
  normalizeTestSuggestionResponse,
  sourceObservationExplanation,
} = require('../../.frontend-test-build/investigation-response.js');
const fixture = JSON.parse(await readFile(new URL('./fixtures/investigation-response.json', import.meta.url), 'utf8'));
const template = await readFile(new URL('../../src/app/app.component.html', import.meta.url), 'utf8');

test('maps actual FastAPI source observation content, SOURCE_FACT badge, location, and valid evidence', () => {
  const { investigation, error } = normalizeInvestigationResponse(fixture);
  assert.equal(error, null);
  const observation = investigation.source_observations[0];
  assert.match(observation.text, /source_observations/);
  assert.equal(observation.classification, 'SOURCE_FACT');
  assert.equal(observation.source_file_path, 'src/app/app.component.ts');
  assert.equal(observation.source_line_range, '11-16');
  assert.equal(sourceObservationExplanation(observation), 'Selected current source lines for the investigation response types.');
  assert.deepEqual(observation.evidence_references, ['E01']);
});

test('maps complete hypotheses and retains validation method and limitations', () => {
  const { investigation } = normalizeInvestigationResponse(fixture);
  const hypothesis = investigation.hypotheses[0];
  assert.match(hypothesis.hypothesis_text, /falsifiable runtime/);
  assert.equal(hypothesis.hypothesis_type, 'BEHAVIORAL_HYPOTHESIS');
  assert.equal(hypothesis.testability, true);
  assert.equal(hypothesis.hypothesis_id, 'HFD365E60');
  assert.match(hypothesis.validation_method, /regression test/);
  assert.match(hypothesis.limitations, /does not prove/);
  assert.deepEqual(hypothesis.evidence_references, ['E01']);
});

test('retains a validated test proposal attached to a hypothesis', () => {
  const response = structuredClone(fixture);
  response.hypotheses[0].test_proposal = {
    purpose: 'Check deterministic grayscale output.', test_framework: 'pytest with NumPy',
    test_code: 'import numpy as np\ndef test_gray():\n    actual = np.array([[76]], dtype=np.uint8)\n    expected = np.array([[76]], dtype=np.uint8)\n    assert np.array_equal(actual, expected)', expected_behavior: 'The output matches expected pixels.',
    hypothesis_tested: response.hypotheses[0].hypothesis_text, hypothesis_id: response.hypotheses[0].hypothesis_id,
    hypothesis_type: 'BEHAVIORAL_HYPOTHESIS', test_type: 'behavioral_regression', test_objectives: [],
    evidence_references: ['E01'], limitations: 'Does not establish original intent.', execution_status: 'NOT_EXECUTED',
  };
  const { investigation } = normalizeInvestigationResponse(response);
  assert.equal(investigation.hypotheses[0].test_proposal.execution_status, 'NOT_EXECUTED');
  assert.deepEqual(investigation.hypotheses[0].test_proposal.evidence_references, ['E01']);
  assert.equal(investigation.hypotheses[0].test_proposal.hypothesis_id, 'HFD365E60');
});

test('uses the correct action for every supported classification', () => {
  const actions = new Map([
    ['BEHAVIORAL_HYPOTHESIS', 'Generate behavioral regression test'],
    ['TECHNICAL_HYPOTHESIS', 'Validate technical assumption'],
    ['DOCUMENTED_ARCHITECTURAL_ROLE', 'Review documented role'],
    ['ARCHITECTURAL_HYPOTHESIS', 'Investigate architecture'],
    ['HISTORICAL_HYPOTHESIS', 'Find historical evidence'],
  ]);
  for (const [hypothesis_type, expected] of actions) {
    const hypothesis = { hypothesis_type, testability: ['BEHAVIORAL_HYPOTHESIS', 'TECHNICAL_HYPOTHESIS'].includes(hypothesis_type) };
    assert.equal(hypothesisActionLabel(hypothesis), expected);
  }
});

test('resolves only existing evidence IDs so the template cannot emit broken anchors', () => {
  const response = structuredClone(fixture);
  response.source_observations[0].evidence_references = ['E99'];
  response.hypotheses[0].evidence_references = ['E99'];
  const mapped = normalizeInvestigationResponse(response);
  assert.deepEqual(mapped.investigation.source_observations[0].evidence_references, []);
  assert.deepEqual(mapped.investigation.hypotheses[0].evidence_references, []);
  assert.equal(mapped.warnings.length, 2);
});

test('handles null, empty, and missing fields with visible fallback content', () => {
  const response = normalizeInvestigationResponse({
    summary: null, source_observations: [null, {}], hypotheses: [null, {}], evidence: null,
    historical_facts: null, contradictions: null, analysis_scope: null, history: null,
  });
  assert.equal(response.error, null);
  assert.equal(response.investigation.source_observations[0].text, 'This source observation is missing its statement in the API response.');
  assert.equal(response.investigation.hypotheses[0].hypothesis_text, 'This hypothesis is missing its statement in the API response.');
  assert.equal(response.investigation.hypotheses[0].hypothesis_type, 'UNKNOWN_CLASSIFICATION');
  assert.equal(response.investigation.hypotheses[0].testability, false);
  assert.ok(response.warnings.length >= 4);
  assert.deepEqual(response.investigation.evidence, []);
});

test('handles empty arrays and rejects an unexpected top-level API response shape', () => {
  const empty = normalizeInvestigationResponse({ source_observations: [], hypotheses: [], evidence: [] });
  assert.equal(empty.investigation.source_observations.length, 0);
  assert.equal(empty.investigation.hypotheses.length, 0);
  assert.equal(normalizeInvestigationResponse(null).investigation, null);
  assert.match(normalizeInvestigationResponse([]).error, /unexpected investigation response/);
});

test('maps test proposals defensively and preserves NOT_EXECUTED status', () => {
  const mapped = normalizeTestSuggestionResponse({
    purpose: 'Check extracted crop pixels.', test_framework: 'pytest', test_code: 'def test_crop(): assert True',
    expected_behavior: 'Pixel values match.', hypothesis_tested: 'Crop values are stable.',
    test_type: 'behavioral_regression', test_objectives: null, evidence_references: ['E01', 'E99'], limitations: null,
  }, ['E01']);
  assert.equal(mapped.suggestion.execution_status, 'NOT_EXECUTED');
  assert.deepEqual(mapped.suggestion.evidence_references, ['E01']);
  assert.deepEqual(mapped.suggestion.test_objectives, []);
  assert.equal(mapped.suggestion.limitations, 'Limitations were not provided.');
  assert.equal(normalizeTestSuggestionResponse('bad', []).suggestion, null);
});

test('Angular template binds observation and hypothesis content, metadata, references, and fallback text', () => {
  assert.match(template, /item\.text \|\| 'This source observation is missing its statement/);
  assert.match(template, /SOURCE_FACT/);
  assert.match(template, /sourceObservationLocation\(item\)/);
  assert.match(template, /sourceObservationExplanation\(item\)/);
  assert.match(template, /hypothesis\.hypothesis_text \|\| 'This hypothesis is missing its statement/);
  assert.match(template, /hypothesis\.hypothesis_type/);
  assert.match(template, /hypothesis\.testability/);
  assert.match(template, /hypothesis\.hypothesis_id/);
  assert.match(template, /hypothesis\.validation_method/);
  assert.match(template, /hypothesis\.limitations/);
  assert.match(template, /\[href\]="'#evidence-'\+evidence\.id"/);
  assert.match(template, /No matching evidence reference is available/);
  assert.match(template, /\*ngIf="activeHypothesis \|\| testSuggestion"/);
});
