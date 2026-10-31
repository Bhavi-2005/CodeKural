import { CommonModule } from '@angular/common';
import { HttpClient, HttpErrorResponse, HttpParams } from '@angular/common/http';
import { Component, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { switchMap } from 'rxjs';
import { hypothesisActionLabel, normalizeInvestigationResponse, normalizeTestSuggestionResponse, sourceObservationExplanation } from './investigation-response';
import type { Claim, Evidence, Hypothesis, Investigation, TestSuggestion, HistoryCommit } from './investigation-response';

interface FileItem { path: string; size: number }
interface BrowseEntry { name: string; path: string; is_repository: boolean }
interface RepositoryInfo { repository_id: string; name: string; default_branch: string | null; current_branch: string | null }

@Component({
  selector: 'ck-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent {
  private readonly http = inject(HttpClient);

  repoPath = '';
  browseOpen = false;
  browseLoading = false;
  browsePath: string | null = null;
  browseParentPath: string | null = null;
  browseRoots: string[] = [];
  browseEntries: BrowseEntry[] = [];
  browseSearch = '';
  browseError = '';
  repo: RepositoryInfo | null = null;
  files: FileItem[] = [];
  fileSearch = '';
  selectedFile = '';
  source = '';
  sourceLines: string[] = [];
  lineNumbers: number[] = [];
  lineCount = 0;
  startLine = 1;
  endLine = 1;
  question = 'Why was this code added?';
  language: 'en' | 'ta' = 'en';
  dependencyDepth = 2;
  history: HistoryCommit[] = [];
  result: Investigation | null = null;
  testSuggestion: TestSuggestion | null = null;
  activeHypothesis: Hypothesis | null = null;
  hypothesisNotice = '';
  loadingAction: 'connect' | 'file' | 'investigation' | 'test' | '' = '';
  working = '';
  error = '';
  lastFailedAction: 'investigation' | 'test' | '' = '';
  aiConfigured = false;
  healthChecked = false;
  sidebarCollapsed = false;
  sidebarOpen = false;
  testCopied = false;

  constructor() {
    this.http.get<{ ai_configured: boolean }>('/api/health').subscribe({
      next: response => { this.aiConfigured = response.ai_configured; this.healthChecked = true; },
      error: () => { this.healthChecked = true; },
    });
  }

  get visibleFiles(): FileItem[] {
    const term = this.fileSearch.trim().toLowerCase();
    return term ? this.files.filter(file => file.path.toLowerCase().includes(term)) : this.files;
  }

  get busy(): boolean { return this.loadingAction !== ''; }
  get lineRangeValid(): boolean {
    return this.startLine >= 1 && this.endLine >= this.startLine && this.endLine <= this.lineCount && this.selectedCount <= 250;
  }
  get selectedCount(): number { return Math.max(0, Number(this.endLine) - Number(this.startLine) + 1); }
  get currentExtension(): string { return this.selectedFile.split('.').pop()?.toUpperCase() ?? 'FILE'; }

  connect(): void {
    this.error = '';
    this.lastFailedAction = '';
    this.loadingAction = 'connect';
    this.working = 'Connecting to repository';
    this.http.post<RepositoryInfo>(
      '/api/repositories/validate', { path: this.repoPath },
    ).subscribe({
      next: repo => this.attachRepository(repo),
      error: error => this.fail(error),
    });
  }

  uploadLocalFolder(event: Event): void {
    const input = event.target as HTMLInputElement;
    const selected = Array.from(input.files ?? []);
    input.value = '';
    if (!selected.length) return;
    this.browseError = '';
    this.lastFailedAction = '';
    this.loadingAction = 'connect';
    this.working = 'Importing the selected folder to this local app';
    const data = new FormData();
    for (const file of selected) data.append('files', file, file.webkitRelativePath || file.name);
    this.http.post<RepositoryInfo>('/api/repositories/import-folder', data).subscribe({
      next: repo => {
        this.browseOpen = false;
        this.attachRepository(repo);
      },
      error: error => {
        this.loadingAction = '';
        this.working = '';
        this.browseError = typeof error.error?.detail === 'string' ? error.error.detail : 'Could not import the selected folder.';
      },
    });
  }

  private attachRepository(repo: RepositoryInfo): void {
    this.http.get<{ files: FileItem[] }>('/api/repositories/files', { params: { repository_id: repo.repository_id } }).subscribe({
      next: response => {
        this.repo = repo;
        this.files = response.files;
        this.fileSearch = '';
        this.selectedFile = '';
        this.source = '';
        this.sourceLines = [];
        this.history = [];
        this.result = null;
        this.testSuggestion = null;
        this.activeHypothesis = null;
        this.hypothesisNotice = '';
        this.loadingAction = '';
        this.working = '';
        this.sidebarOpen = false;
      },
      error: error => this.fail(error),
    });
  }

  openRepositoryPicker(): void {
    this.browseOpen = true;
    this.browseSearch = '';
    this.browseError = '';
    this.loadBrowse();
  }

  loadBrowse(path?: string, search = ''): void {
    this.browseLoading = true;
    this.browseError = '';
    let params = new HttpParams();
    if (path) params = params.set('path', path);
    if (search.trim()) params = params.set('search', search.trim());
    this.http.get<{ roots: string[]; current_path: string | null; parent_path?: string | null; entries: BrowseEntry[] }>('/api/repositories/browse', { params })
      .subscribe({
        next: response => {
          this.browseRoots = response.roots;
          this.browsePath = response.current_path;
          this.browseParentPath = response.parent_path ?? null;
          this.browseEntries = response.entries;
          this.browseLoading = false;
        },
        error: error => {
          this.browseLoading = false;
          this.browseError = typeof error.error?.detail === 'string' ? error.error.detail : 'Could not browse local folders. Check the backend WORKSPACE_ROOT setting.';
        },
      });
  }

  searchRepositories(): void {
    this.loadBrowse(undefined, this.browseSearch);
  }

  pickBrowseEntry(entry: BrowseEntry): void {
    if (entry.is_repository) {
      this.repoPath = entry.path;
      this.browseOpen = false;
      this.connect();
    } else {
      this.loadBrowse(entry.path);
    }
  }

  selectFile(path: string): void {
    if (!this.repo || this.busy) return;
    this.selectedFile = path;
    this.error = '';
    this.lastFailedAction = '';
    this.result = null;
    this.testSuggestion = null;
    this.activeHypothesis = null;
    this.hypothesisNotice = '';
    this.history = [];
    this.source = '';
    this.sourceLines = [];
    this.loadingAction = 'file';
    this.working = 'Reading source and history';
    const params = new HttpParams().set('repository_id', this.repo.repository_id).set('file_path', path);
    this.http.get<{ commits: HistoryCommit[]; blame: string[]; diffs: unknown[] }>('/api/repositories/history', { params })
      .pipe(switchMap(history => {
        this.history = history.commits;
        return this.http.get<{ content: string; line_count: number }>('/api/repositories/source', { params });
      }))
      .subscribe({
        next: source => {
          this.source = source.content;
          this.sourceLines = source.content.split('\n');
          this.lineCount = source.line_count;
          this.startLine = 1;
          this.endLine = Math.min(source.line_count, 40);
          this.updateCodeWindow();
          this.loadingAction = '';
          this.working = '';
          this.sidebarOpen = false;
        },
        error: error => this.fail(error),
      });
  }

  newInvestigation(): void {
    this.result = null;
    this.testSuggestion = null;
    this.activeHypothesis = null;
    this.hypothesisNotice = '';
    this.error = '';
    this.lastFailedAction = '';
    this.testCopied = false;
    this.question = 'Why was this code added?';
    this.sidebarOpen = false;
  }

  investigate(): void {
    if (!this.repo || !this.selectedFile || this.busy || !this.aiConfigured) return;
    if (!this.lineRangeValid) {
      this.error = 'Select a valid range of up to 250 lines within the file.';
      return;
    }
    if (this.question.trim().length < 3) {
      this.error = 'Ask a question of at least three characters.';
      return;
    }
    this.error = '';
    this.loadingAction = 'investigation';
    this.working = 'Reviewing repository evidence with Sarvam';
    this.http.post<unknown>('/api/investigations', this.payload()).subscribe({
      next: response => {
        const normalized = normalizeInvestigationResponse(response);
        if (!normalized.investigation) {
          this.result = null;
          this.testSuggestion = null;
          this.activeHypothesis = null;
          this.hypothesisNotice = '';
          this.error = normalized.error ?? 'The backend returned an invalid investigation response.';
          this.lastFailedAction = 'investigation';
        } else {
          this.result = normalized.investigation;
          this.testSuggestion = null;
          this.activeHypothesis = null;
          this.hypothesisNotice = '';
          this.lastFailedAction = '';
        }
        this.loadingAction = '';
        this.working = '';
      },
      error: error => this.fail(error),
    });
  }

  hypothesisAction(hypothesis: Hypothesis): string {
    return hypothesisActionLabel(hypothesis);
  }

  sourceObservationExplanation(item: Claim): string { return sourceObservationExplanation(item); }

  sourceObservationLocation(item: Claim): string {
    if (item.source_file_path && item.source_line_range) return `${item.source_file_path} · lines ${item.source_line_range}`;
    if (item.source_file_path) return item.source_file_path;
    return 'Source location unavailable';
  }

  selectHypothesis(hypothesis: Hypothesis): void {
    this.activeHypothesis = hypothesis;
    this.testSuggestion = null;
    this.hypothesisNotice = '';
    if (!hypothesis.testability || (hypothesis.hypothesis_type !== 'BEHAVIORAL_HYPOTHESIS' && hypothesis.hypothesis_type !== 'TECHNICAL_HYPOTHESIS')) {
      this.hypothesisNotice = hypothesis.hypothesis_type === 'DOCUMENTED_ARCHITECTURAL_ROLE'
        ? 'The cited documentation describes this module role. It does not establish original historical intent or runtime behavior.'
        : 'This hypothesis cannot be established through a regression test. Inspect Git history, related modules, design documents, pull requests, or issues as relevant.';
      return;
    }
    if (hypothesis.test_proposal) {
      this.testSuggestion = hypothesis.test_proposal;
      this.testCopied = false;
      return;
    }
    this.suggestTest();
  }

  suggestTest(): void {
    if (!this.repo || !this.selectedFile || !this.result || this.busy || !this.aiConfigured) return;
    if (!this.activeHypothesis) return;
    this.error = '';
    this.loadingAction = 'test';
    this.working = 'Drafting a regression test';
    this.http.post<unknown>('/api/investigations/test-suggestion', {
      investigation: this.payload(),
      hypothesis: this.activeHypothesis,
    }).subscribe({
      next: response => {
        const normalized = normalizeTestSuggestionResponse(response, (this.result?.evidence ?? []).map(item => item.id));
        if (!normalized.suggestion) {
          this.testSuggestion = null;
          this.error = normalized.error ?? 'The backend returned an invalid test proposal.';
          this.lastFailedAction = 'test';
        } else {
          this.testSuggestion = normalized.suggestion;
          this.lastFailedAction = '';
          this.testCopied = false;
        }
        this.loadingAction = '';
        this.working = '';
      },
      error: error => this.fail(error),
    });
  }

  languageChanged(): void {
    if (this.result && !this.busy) this.investigate();
  }

  retryFailedAction(): void {
    if (this.lastFailedAction === 'test') this.suggestTest();
    else this.investigate();
  }

  investigationInputsChanged(): void {
    if (!this.result) return;
    this.result = null;
    this.testSuggestion = null;
    this.activeHypothesis = null;
    this.hypothesisNotice = '';
    this.activeHypothesis = null;
    this.hypothesisNotice = '';
    this.testCopied = false;
    this.error = '';
    this.lastFailedAction = '';
  }

  updateCodeWindow(): void {
    this.investigationInputsChanged();
    const start = Math.max(1, Math.min(Number(this.startLine) || 1, this.lineCount || 1));
    const end = Math.max(start, Math.min(Number(this.endLine) || start, this.lineCount || 1));
    const first = Math.max(1, start - 100);
    const last = Math.min(this.lineCount, Math.max(end + 40, first + 759));
    const windowStart = Math.max(1, Math.min(first, last - 799));
    this.lineNumbers = Array.from({ length: Math.max(0, last - windowStart + 1) }, (_, index) => windowStart + index);
  }

  evidenceFor(ids: string[] | null | undefined): Evidence[] {
    const references = Array.isArray(ids) ? new Set(ids) : new Set<string>();
    return (this.result?.evidence ?? []).filter(item => references.has(item.id));
  }

  evidenceGroups(): { category: string; items: Evidence[] }[] {
    const groups = new Map<string, Evidence[]>();
    for (const item of this.result?.evidence ?? []) groups.set(item.category, [...(groups.get(item.category) ?? []), item]);
    return [...groups.entries()].map(([category, items]) => ({ category, items }));
  }

  async copyTest(): Promise<void> {
    if (!this.testSuggestion) return;
    try {
      await navigator.clipboard.writeText(this.testSuggestion.test_code);
      this.testCopied = true;
    } catch {
      this.error = 'Clipboard access was unavailable. Select and copy the test code instead.';
    }
  }

  private payload() {
    return {
      repository_id: this.repo!.repository_id,
      file_path: this.selectedFile,
      start_line: Number(this.startLine),
      end_line: Number(this.endLine),
      question: this.question.trim(),
      language: this.language,
      dependency_depth: Number(this.dependencyDepth),
    };
  }

  private fail(error: HttpErrorResponse): void {
    if (this.loadingAction === 'investigation' || this.loadingAction === 'test') {
      this.lastFailedAction = this.loadingAction;
    }
    this.loadingAction = '';
    this.working = '';
    const detail = error.error?.detail;
    if (typeof detail === 'string') this.error = detail;
    else if (Array.isArray(detail)) {
      this.error = detail.map(item => typeof item?.msg === 'string' ? item.msg : '').filter(Boolean).join(' ')
        || 'The request was invalid. Check the repository path and selected line range.';
    } else {
      this.error = 'The request could not be completed. Check the repository path and API status, then retry.';
    }
  }
}
