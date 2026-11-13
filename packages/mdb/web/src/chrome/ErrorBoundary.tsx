// Contain a failing view (including an EvidenceMissing throw) so the rest of the app keeps working.
import { Component, type ReactNode } from 'react';

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<{ name: string; children: ReactNode }, State> {
  override state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  override render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="na" role="alert">
        <strong>{this.props.name} failed:</strong> {this.state.error.message}{' '}
        <button type="button" onClick={() => this.setState({ error: null })}>
          retry
        </button>
      </div>
    );
  }
}
