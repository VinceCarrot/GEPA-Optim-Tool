// Thin fetch wrapper around the FastAPI backend (/api/*).

function errorText(detail, status) {
  if (!detail) return `HTTP ${status}`;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => (d.loc ? `${d.loc}: ${d.msg}` : d.msg || JSON.stringify(d))).join("; ");
  return JSON.stringify(detail);
}

async function request(method, url, body) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) {
    const err = new Error(errorText(data && data.detail !== undefined ? data.detail : data, res.status));
    err.status = res.status;
    throw err;
  }
  return data;
}

export const api = {
  get: (u) => request("GET", u),
  post: (u, b) => request("POST", u, b === undefined ? {} : b),
  put: (u, b) => request("PUT", u, b),
  upload: (u, file) => {
    const fd = new FormData();
    fd.append("file", file);
    return request("POST", u, fd);
  },
};
