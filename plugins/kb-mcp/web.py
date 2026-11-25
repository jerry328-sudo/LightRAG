"""Browser pages for consent, citations and explicit administrative approval."""

from html import escape

from starlette.responses import HTMLResponse

PAGE_POLICY = (
    "default-src 'none'; style-src 'unsafe-inline'; form-action {form_action}; "
    "base-uri 'none'; frame-ancestors 'none'"
)
HEADERS = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "same-origin",
    "Content-Security-Policy": PAGE_POLICY.format(form_action="'self'"),
    "X-Content-Type-Options": "nosniff",
}


def page(title, body, status=200):
    return HTMLResponse(
        f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title>
<style>
:root{{color-scheme:light dark}}body{{font:1rem/1.55 system-ui,sans-serif;
max-width:38rem;margin:8vh auto;padding:0 1.25rem}}
label{{display:block;margin:1rem 0}}input,button{{font:inherit}}
input{{width:100%;box-sizing:border-box;padding:.5rem}}button{{padding:.5rem 1rem}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
</style><main><h1>{escape(title)}</h1>{body}</main></html>""",
        status_code=status,
        headers=HEADERS,
    )


def credentials(title, detail, ticket, action, button, *, callback_origin=None):
    response = page(
        title,
        detail + f"""<form method="post" action="{escape(action, quote=True)}">
<input type="hidden" name="ticket" value="{escape(ticket, quote=True)}">
<label>用户名<input name="username" maxlength="128" autocomplete="username" required></label>
<label>密码<input name="password" type="password" maxlength="512" autocomplete="current-password" required></label>
<button type="submit">{escape(button)}</button></form>""",
    )
    if callback_origin is not None:
        # Chromium applies the form's CSP to the 303 callback as well as the POST.
        response.headers["Content-Security-Policy"] = PAGE_POLICY.format(
            form_action=f"'self' {callback_origin}"
        )
    return response
