export interface Evidence { id: string; type: string; category: string; location: string; path?: string; commit?: string; line_range?: string; url?: string; excerpt: string; relevance_reason: string }
export interface Claim { text: string; evidence_references: string[]; hypothesis_type?: string | null; classification?: string; source_file_path?: string | null; source_line_range?: string | null; explanation?: string }
export type HypothesisType = 'BEHAVIORAL_HYPOTHESIS' | 'TECHNICAL_HYPOTHESIS' | 'DOCUMENTED_ARCHITECTURAL_ROLE' | 'ARCHITECTURAL_HYPOTHESIS' | 'HISTORICAL_HYPOTHESIS' | 'UNKNOWN_CLASSIFICATION';
export interface Hypothesis { hypothesis_id: string; hypothesis_text: string; hypothesis_type: HypothesisType; testability: boolean; evidence_references: string[]; validation_method: string; limitations: string; test_proposal?: TestSuggestion }
export interface Contradiction { description: string; why_it_matters: string; evidence_references: string[] }
export interface TestDiscovery { total_test_files_found: number; references: { test_id: string; path: string; test_names: string[]; line_ranges: string[]; classification: 'DIRECT' | 'INDIRECT' | 'UNRELATED' | 'UNKNOWN'; relationship_reason: string; evidence_references: string[] }[]; counts: Record<'DIRECT' | 'INDIRECT' | 'UNRELATED' | 'UNKNOWN', number>; coverage_status: 'measured' | 'not_measured'; coverage_percent: number | null; coverage_report_path: string | null; coverage_unknown: boolean }
export interface Investigation { summary: string; summary_evidence_references: string[]; source_observations: (Claim & { classification?: 'SOURCE_FACT' })[]; historical_facts: Claim[]; hypotheses: Hypothesis[]; contradictions: Contradiction[]; unknowns: string[]; evidence_status: string; history_status: string; render_warnings: string[]; analysis_scope: { files_available: number; files_inspected: number; complete: boolean; evidence_complete: boolean; files_skipped_count: number; related_tests_found: boolean; test_discovery: TestDiscovery; symbols_traced: string[]; scope_description: string; dependency_depth: number; inspected_paths: string[]; files_skipped: {path: string; reason: string}[]; category_counts: Record<string, number>; note?: string }; evidence: Evidence[]; history: { commit_count: number; shallow: boolean; empty: boolean; insufficient: boolean; commits: HistoryCommit[]; line_trace: unknown[] } }
export interface HistoryCommit { hash: string; short_hash: string; subject: string; author: string; date: string; url?: string }
export interface TestObjective { test_type: string; description: string; testable: boolean; evidence_references: string[] }
export interface TestSuggestion { purpose: string; test_framework: string; test_code: string; expected_behavior: string; hypothesis_tested: string; hypothesis_id?: string | null; hypothesis_type?: 'BEHAVIORAL_HYPOTHESIS' | 'TECHNICAL_HYPOTHESIS' | null; test_type: string; test_objectives: TestObjective[]; evidence_references: string[]; limitations: string; execution_status: 'NOT_EXECUTED' }

const HYPOTHESIS_TYPES = new Set<HypothesisType>([
  'BEHAVIORAL_HYPOTHESIS', 'TECHNICAL_HYPOTHESIS', 'DOCUMENTED_ARCHITECTURAL_ROLE',
  'ARCHITECTURAL_HYPOTHESIS', 'HISTORICAL_HYPOTHESIS',
]);

function object(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
}

function string(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? [...new Set(value.filter((item): item is string => typeof item === 'string' && !!item.trim()).map(item => item.trim()))] : [];
}

function references(item: Record<string, unknown>, evidence: Evidence[]): string[] {
  const ids = strings(item['evidence_references'] ?? item['evidenceReferences']);
  const existing = new Set(evidence.map(record => record.id));
  return ids.filter(id => existing.has(id));
}

function normalizeEvidence(value: unknown): Evidence[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap(raw => {
    const item = object(raw);
    const id = item && string(item['id']);
    if (!item || !id || !/^E\d{2}$/.test(id)) return [];
    return [{
      id,
      type: string(item['type']) ?? 'unknown',
      category: string(item['category']) ?? 'A_source',
      location: string(item['location']) ?? string(item['path']) ?? 'Repository location unavailable',
      path: string(item['path']) ?? undefined,
      line_range: string(item['line_range']) ?? undefined,
      url: string(item['url']) ?? undefined,
      excerpt: string(item['excerpt']) ?? '',
      relevance_reason: string(item['relevance_reason']) ?? 'Evidence was returned without an explanation.',
    }];
  });
}

function sourceObservation(raw: unknown, index: number, evidence: Evidence[], warnings: string[]): Claim & {
  classification: 'SOURCE_FACT'; source_file_path: string | null; source_line_range: string | null; explanation: string;
} {
  const item = object(raw);
  const text = item && (string(item['text']) ?? string(item['observation']) ?? string(item['statement']));
  if (!item || !text) warnings.push(`Source observation ${index + 1} is missing its statement; showing a fallback.`);
  const evidenceReferences = item ? references(item, evidence) : [];
  if (item && strings(item['evidence_references'] ?? item['evidenceReferences']).length !== evidenceReferences.length) {
    warnings.push(`Source observation ${index + 1} references evidence that is not in this response.`);
  }
  const sourceEvidence = evidenceReferences.map(id => evidence.find(record => record.id === id)!)
    .find(record => record.type === 'source' || record.type === 'source_function')
    ?? evidenceReferences.map(id => evidence.find(record => record.id === id)!).find(Boolean);
  return {
    text: text ?? 'This source observation is missing its statement in the API response.',
    classification: 'SOURCE_FACT',
    evidence_references: evidenceReferences,
    source_file_path: sourceEvidence?.path ?? null,
    source_line_range: sourceEvidence?.line_range ?? null,
    explanation: item ? string(item['explanation']) ?? sourceEvidence?.relevance_reason ?? 'Observed in the current source included with this investigation.'
      : 'The backend returned an unexpected source observation shape.',
  };
}

function hypothesis(raw: unknown, index: number, evidence: Evidence[], warnings: string[]): Hypothesis {
  const item = object(raw);
  const text = item && (string(item['hypothesis_text']) ?? string(item['text']) ?? string(item['statement']));
  if (!item || !text) warnings.push(`Hypothesis ${index + 1} is missing its statement; showing a fallback.`);
  const typeValue = item && (string(item['hypothesis_type']) ?? string(item['classification']));
  const type: HypothesisType = typeValue && HYPOTHESIS_TYPES.has(typeValue as HypothesisType)
    ? typeValue as HypothesisType : 'UNKNOWN_CLASSIFICATION';
  if (item && type === 'UNKNOWN_CLASSIFICATION') warnings.push(`Hypothesis ${index + 1} has an unknown classification.`);
  const evidenceReferences = item ? references(item, evidence) : [];
  if (item && strings(item['evidence_references'] ?? item['evidenceReferences']).length !== evidenceReferences.length) {
    warnings.push(`Hypothesis ${index + 1} references evidence that is not in this response.`);
  }
  const testable = type === 'BEHAVIORAL_HYPOTHESIS' || type === 'TECHNICAL_HYPOTHESIS';
  const proposal = item && object(item['test_proposal'])
    ? normalizeTestSuggestionResponse(item['test_proposal'], evidence.map(record => record.id)).suggestion ?? undefined
    : undefined;
  if (testable && item && item['test_proposal'] && !proposal) warnings.push(`Hypothesis ${index + 1} has an invalid test proposal.`);
  return {
    hypothesis_id: item ? string(item['hypothesis_id']) ?? 'ID unavailable' : 'ID unavailable',
    hypothesis_text: text ?? 'This hypothesis is missing its statement in the API response.',
    hypothesis_type: type,
    testability: testable,
    evidence_references: evidenceReferences,
    validation_method: item ? string(item['validation_method']) ?? 'Validation method was not provided.' : 'The backend returned an unexpected hypothesis shape.',
    limitations: item ? string(item['limitations']) ?? 'Limitations were not provided.' : 'Review the API response before relying on this hypothesis.',
    test_proposal: proposal,
  };
}

function normalizeTestDiscovery(raw: unknown) {
  const discovery = object(raw) ?? {};
  const refs = Array.isArray(discovery['references']) ? discovery['references'] : null;
  const canonical = refs ? refs.flatMap((value, index) => {
    const item = object(value);
    const path = item && string(item['path']);
    if (!item || !path) return [];
    const testNames = strings(item['test_names']);
    const lineRanges = strings(item['line_ranges']);
    const classification = string(item['classification']);
    const known = ['DIRECT', 'INDIRECT', 'UNRELATED', 'UNKNOWN'].includes(classification ?? '');
    return [{
      test_id: string(item['test_id']) ?? `${path}::${testNames[0] ?? 'file'}#${lineRanges[0] ?? index}`,
      path, test_names: testNames, line_ranges: lineRanges,
      classification: known ? classification as 'DIRECT' | 'INDIRECT' | 'UNRELATED' | 'UNKNOWN' : 'UNKNOWN',
      relationship_reason: string(item['relationship_reason']) ?? 'Relationship could not be determined.',
      evidence_references: strings(item['evidence_references']),
    }];
  }) : [
    ...(['directly_related_tests', 'indirectly_related_tests', 'unrelated_tests'] as const).flatMap((key, categoryIndex) =>
      (Array.isArray(discovery[key]) ? discovery[key] as unknown[] : []).flatMap((value, index) => {
        const item = object(value);
        const path = item && string(item['path']);
        if (!item || !path) return [];
        const testNames = strings(item['test_names']);
        const lineRanges = strings(item['line_ranges']);
        return [{ test_id: `${path}::${testNames[0] ?? 'file'}#${lineRanges[0] ?? index}`,
          path, test_names: testNames, line_ranges: lineRanges,
          classification: (['DIRECT', 'INDIRECT', 'UNRELATED'][categoryIndex]) as 'DIRECT' | 'INDIRECT' | 'UNRELATED',
          relationship_reason: 'Classified by the backend test discovery response.',
          evidence_references: strings(item['evidence_references']) }];
      })),
    ...(Array.isArray(discovery['uninspected_tests']) ? discovery['uninspected_tests'] as unknown[] : []).flatMap(pathValue => {
      const path = string(pathValue);
      return path ? [{ test_id: `${path}::file#unknown`, path, test_names: [], line_ranges: [],
        classification: 'UNKNOWN' as const, relationship_reason: 'Test file could not be inspected.', evidence_references: [] }] : [];
    }),
  ];
  const categories = ['DIRECT', 'INDIRECT', 'UNRELATED', 'UNKNOWN'] as const;
  const counts = Object.fromEntries(categories.map(category => [category,
    canonical.filter(item => item.classification === category).length])) as Record<typeof categories[number], number>;
  const measured = discovery['coverage_status'] === 'measured' && typeof discovery['coverage_percent'] === 'number';
  return {
    total_test_files_found: new Set(canonical.map(item => item.path)).size,
    references: canonical, counts,
    coverage_status: measured ? 'measured' as const : 'not_measured' as const,
    coverage_percent: measured ? discovery['coverage_percent'] as number : null,
    coverage_report_path: measured ? string(discovery['coverage_report_path']) : null,
    coverage_unknown: !measured,
  };
}

export function normalizeInvestigationResponse(raw: unknown): { investigation: Investigation | null; warnings: string[]; error: string | null } {
  const payload = object(raw);
  if (!payload) return { investigation: null, warnings: [], error: 'The backend returned an unexpected investigation response. Please retry.' };
  const warnings: string[] = [];
  const evidence = normalizeEvidence(payload['evidence']);
  const normalizeClaims = (key: 'historical_facts' | 'contradictions'): unknown[] => Array.isArray(payload[key]) ? payload[key] as unknown[] : [];
  if (!Array.isArray(payload['source_observations'])) warnings.push('The response did not include a source observation list.');
  if (!Array.isArray(payload['hypotheses'])) warnings.push('The response did not include a hypothesis list.');
  const sourceObservations = (Array.isArray(payload['source_observations']) ? payload['source_observations'] : [])
    .map((item, index) => sourceObservation(item, index, evidence, warnings));
  const hypotheses = (Array.isArray(payload['hypotheses']) ? payload['hypotheses'] : [])
    .map((item, index) => hypothesis(item, index, evidence, warnings));
  const scope = object(payload['analysis_scope']) ?? {};
  const history = object(payload['history']) ?? {};
  const investigation = {
    ...payload,
    summary: string(payload['summary']) ?? 'The backend did not return an investigation summary.',
    summary_evidence_references: references(object({ evidence_references: payload['summary_evidence_references'] })!, evidence),
    source_observations: sourceObservations,
    historical_facts: normalizeClaims('historical_facts') as Claim[],
    hypotheses,
    contradictions: normalizeClaims('contradictions') as Investigation['contradictions'],
    unknowns: strings(payload['unknowns']),
    evidence_status: string(payload['evidence_status']) ?? 'partial',
    history_status: string(payload['history_status']) ?? 'unknown',
    analysis_scope: {
      ...scope,
      files_available: typeof scope['files_available'] === 'number' ? scope['files_available'] : 0,
      files_inspected: typeof scope['files_inspected'] === 'number' ? scope['files_inspected'] : 0,
      complete: scope['complete'] === true,
      evidence_complete: scope['evidence_complete'] !== false,
      files_skipped_count: typeof scope['files_skipped_count'] === 'number' ? scope['files_skipped_count'] : 0,
      related_tests_found: scope['related_tests_found'] === true,
      test_discovery: normalizeTestDiscovery(scope['test_discovery']),
      symbols_traced: strings(scope['symbols_traced']),
      scope_description: string(scope['scope_description']) ?? 'Repository scan details are unavailable.',
      dependency_depth: typeof scope['dependency_depth'] === 'number' ? scope['dependency_depth'] : 0,
      inspected_paths: strings(scope['inspected_paths']),
      files_skipped: Array.isArray(scope['files_skipped']) ? scope['files_skipped'] : [],
      category_counts: object(scope['category_counts']) ?? {},
    },
    evidence,
    history: {
      ...history,
      commit_count: typeof history['commit_count'] === 'number' ? history['commit_count'] : 0,
      shallow: history['shallow'] === true,
      empty: history['empty'] !== false,
      insufficient: history['insufficient'] === true,
      commits: Array.isArray(history['commits']) ? history['commits'] : [],
      line_trace: Array.isArray(history['line_trace']) ? history['line_trace'] : [],
    },
    render_warnings: warnings,
  } as Investigation;
  return { investigation, warnings, error: null };
}

export function sourceObservationExplanation(item: Claim): string {
  return string((item as Claim & { explanation?: unknown }).explanation) ?? 'Observed in the current source included with this investigation.';
}

export function normalizeTestSuggestionResponse(raw: unknown, knownEvidenceIds: string[]): { suggestion: TestSuggestion | null; error: string | null } {
  const item = object(raw);
  if (!item) return { suggestion: null, error: 'The backend returned an unexpected test proposal. Please retry.' };
  const known = new Set(knownEvidenceIds);
  const cleanReferences = (value: unknown) => strings(value).filter(id => known.has(id));
  const testObjectives = Array.isArray(item['test_objectives']) ? item['test_objectives'].flatMap(rawObjective => {
    const objective = object(rawObjective);
    if (!objective) return [];
    return [{ test_type: string(objective['test_type']) ?? 'hypothesis_validation',
      description: string(objective['description']) ?? 'Objective details were not provided.',
      testable: objective['testable'] === true,
      evidence_references: cleanReferences(objective['evidence_references']) }];
  }) : [];
  return { suggestion: {
    purpose: string(item['purpose']) ?? 'Purpose was not provided.',
    test_framework: string(item['test_framework']) ?? 'Framework unavailable',
    test_code: string(item['test_code']) ?? 'Test code was not provided.',
    expected_behavior: string(item['expected_behavior']) ?? 'Expected behavior was not provided.',
    hypothesis_tested: string(item['hypothesis_tested']) ?? 'Hypothesis details were not provided.',
    hypothesis_id: string(item['hypothesis_id']),
    hypothesis_type: item['hypothesis_type'] === 'BEHAVIORAL_HYPOTHESIS' || item['hypothesis_type'] === 'TECHNICAL_HYPOTHESIS'
      ? item['hypothesis_type'] : null,
    test_type: string(item['test_type']) ?? 'hypothesis_validation',
    test_objectives: testObjectives,
    evidence_references: cleanReferences(item['evidence_references']),
    limitations: string(item['limitations']) ?? 'Limitations were not provided.',
    execution_status: 'NOT_EXECUTED',
  }, error: null };
}

export function hypothesisActionLabel(item: Hypothesis): string {
  if (item.testability && item.hypothesis_type === 'BEHAVIORAL_HYPOTHESIS') return 'Generate behavioral regression test';
  if (item.testability && item.hypothesis_type === 'TECHNICAL_HYPOTHESIS') return 'Validate technical assumption';
  if (item.hypothesis_type === 'DOCUMENTED_ARCHITECTURAL_ROLE') return 'Review documented role';
  if (item.hypothesis_type === 'ARCHITECTURAL_HYPOTHESIS') return 'Investigate architecture';
  if (item.hypothesis_type === 'HISTORICAL_HYPOTHESIS') return 'Find historical evidence';
  return 'Review investigation evidence';
}
