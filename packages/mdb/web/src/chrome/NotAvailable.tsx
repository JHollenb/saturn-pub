// Shown wherever the project does not hold what a view needs. Always says why.
import type { Refused } from '../api/types.gen';

export function NotAvailable({ what, reason }: { what: string; reason?: string | null }) {
  return (
    <div className="na" role="note" data-testid="not-available">
      <strong>{what}: not available.</strong> {reason ?? 'The project holds no data for this.'}
    </div>
  );
}

/** Plain-language text for the refusal codes the server envelope can carry. The code itself is
 *  always shown beside it; unknown codes show the code and the server's detail only. */
export const REFUSAL_TEXT: Record<string, string> = {
  host_not_allowed:
    'The server does not accept this Host header; open it at the address it printed (localhost or 127.0.0.1).',
  origin_not_allowed: 'The server does not accept requests from this page origin.',
  over_limit: 'The request exceeds a server limit (size, count or rate); narrow it and retry.',
  over_cap: 'The answer would exceed the response cap, so the server sent nothing.',
  missing_host: 'The request carried no Host header, so the server refused it.',
  method_not_allowed: 'The server does not allow this HTTP method on that route (it is read-only).',
  bad_request: 'The server could not parse the request.',
  http_error: 'The server answered with an HTTP error it could not describe further.',
  internal_error: 'The server failed on this request; its log has the detail.',
  bad_params: 'The server could not use the request parameters.',
  web_build_absent: 'The web build is not installed on the server, so there is no page to show.',
  bad_envelope: 'The server answered with a body this page cannot read.',
  bad_response: 'The server answer was not JSON.',
  network: 'The server could not be reached.',
};

export function RefusedView({ what, refused }: { what: string; refused: Refused }) {
  return (
    <div className="na" role="note" data-refused={refused.refused}>
      <strong>{what}: refused</strong> <code>{refused.refused}</code>
      {REFUSAL_TEXT[refused.refused] ? (
        <span className="muted"> {REFUSAL_TEXT[refused.refused]}</span>
      ) : null}
      {refused.detail ? <span className="muted"> {refused.detail}</span> : null}
      {refused.bytes != null ? (
        <span className="faint">
          {' '}
          ({String(refused.bytes)} bytes{refused.cap != null ? `, cap ${String(refused.cap)}` : ''})
        </span>
      ) : null}
    </div>
  );
}
