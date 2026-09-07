// Receives one rater's completed sheet and stores it as a JSON blob.
//
// The scorer (src/human_eval.py) reads {rater, ratings:[{query_index, rank,
// clip_id, rating}]}, so that shape is validated here and stored verbatim --
// this endpoint never invents, defaults or repairs a rating. A clip the rater
// could not listen to arrives as null and stays null; the scorer skips nulls
// rather than counting them as a low score.

import { put } from '@vercel/blob';

const MAX_BODY = 64 * 1024;

function invalid(body) {
  if (!body || typeof body !== 'object') return 'body must be an object';
  if (typeof body.rater !== 'string' || !body.rater.trim()) return 'rater is required';
  if (body.rater.length > 80) return 'rater name is too long';
  if (!Array.isArray(body.ratings) || body.ratings.length === 0) return 'ratings is required';
  if (body.ratings.length > 200) return 'too many ratings';
  for (const r of body.ratings) {
    if (!r || typeof r !== 'object') return 'each rating must be an object';
    if (!Number.isInteger(r.query_index) || r.query_index < 0) return 'bad query_index';
    if (!Number.isInteger(r.rank) || r.rank < 1) return 'bad rank';
    if (r.rating !== null && !(Number.isInteger(r.rating) && r.rating >= 1 && r.rating <= 5)) {
      return 'rating must be an integer 1-5, or null when the rater could not listen';
    }
  }
  return null;
}

function slug(name) {
  return name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40)
    || 'rater';
}

export default async function handler(req, res) {
  // A file:// copy of the sheet posts from a null origin, so allow it: the
  // endpoint stores ratings for a public study, holds nothing private, and
  // validates every field it accepts.
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  if (req.method === 'OPTIONS') return res.status(204).end();
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'use POST' });
  }

  let body = req.body;
  if (typeof body === 'string') {
    if (body.length > MAX_BODY) return res.status(413).json({ error: 'body too large' });
    try { body = JSON.parse(body); } catch { return res.status(400).json({ error: 'invalid JSON' }); }
  }

  const problem = invalid(body);
  if (problem) return res.status(400).json({ error: problem });

  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const path = `ratings/${slug(body.rater)}-${stamp}.json`;

  try {
    // addRandomSuffix keeps a second submission from the same person under the
    // same name from silently overwriting the first -- the researcher decides
    // which to keep, not this endpoint.
    const blob = await put(path, JSON.stringify(body, null, 2), {
      access: 'public',
      addRandomSuffix: true,
      contentType: 'application/json',
    });
    const answered = body.ratings.filter((r) => r.rating !== null).length;
    return res.status(200).json({ ok: true, stored: blob.pathname, answered });
  } catch (err) {
    console.error('blob put failed', err);
    return res.status(500).json({ error: 'could not store ratings' });
  }
}
