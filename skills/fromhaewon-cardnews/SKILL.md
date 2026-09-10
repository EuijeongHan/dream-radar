---
name: fromhaewon-cardnews
description: Select, verify, design, caption, archive, and prepare daily AI-paper Instagram card news for @fromhaewon and 한해원의 관측소. Use for paper-topic selection, PAPER RADAR cards, captions, hashtags, posting, or the associated development-insight ledger.
---

# 한해원의 관측소 카드뉴스

Produce a trustworthy, recognizable daily AI-paper briefing for `@fromhaewon`. Optimize for useful discovery and repeatable quality, not hype alone.

## Start here

Before each related task:

1. Inspect existing entries under `cards/` and the research ledger to avoid repeating a paper or near-identical topic.
2. Preserve the established brand and numbering sequence.
3. Distinguish verified paper claims from Radar's interpretation and future product ideas.

## Select a paper

- Research current AI attention using fresh evidence: recent news, credible research discussion, new-paper momentum, practical relevance, and likely interest among Korean AI practitioners.
- Prefer a strong intersection of timeliness, technical substance, explainability in five cards, and relevance to building models or agents.
- Do not select solely because a keyword is trending. Reject weak, unverifiable, promotional, duplicated, or legally unsuitable material.
- Prefer the primary paper page and official project or code repositories. Record the title, all authors, submission or publication date, canonical URL, and evidence reviewed.
- Check whether an accessible implementation, benchmark, dataset, or evaluation method could be useful to Radar. Verify its actual repository, license, maturity, security implications, and maintenance status before recommending it.

## Verify the summary

- Read the full paper when available. If only the abstract was reviewed, say so on the card and in the caption.
- Tie every substantive statement, comparison, and number to the reviewed source. Preserve experimental scope and conditions; never turn correlation or benchmark improvement into a universal claim.
- Label Radar's application ideas as interpretation, experiment, or hypothesis rather than paper findings.
- When confidence is insufficient, omit the claim or explicitly state what requires checking in the full paper.

## Design system

- Create five portrait cards at `1080 × 1350` pixels.
- Maintain the editorial palette: warm cream, deep forest green, and restrained vermilion red; muted gold may be used sparingly.
- Use clean Korean typography, strong hierarchy, generous margins, and legible mobile-sized text. Avoid dense paper-like layouts and decorative clutter.
- Keep this identity on the cover: `FROM HAEWON · PAPER RADAR NNN` and `논문 한 편, 5장으로 읽기`.
- Recommended narrative: cover and question → problem → core method → how it works or key mechanism → result, limitations, and source.
- Use original diagrams and visual metaphors. Do not reuse a paper's figures, tables, screenshots, or substantial text unless its license clearly permits the intended reuse and attribution is supplied.
- The final card must show the full paper title, author attribution or a clear pointer to it, the canonical source URL or link instruction, and the evidence-basis notice such as `논문 초록 기반 요약` or `논문 본문 기반 요약`.
- Put `© fromhaewon. All rights reserved.` in a readable but unobtrusive footer on the final card. This marks the original summary and card design; it must not imply ownership of the underlying paper.

Save each issue under `cards/<paper-slug>/` with numbered source and rendered files plus `caption.md`. Visually inspect all rendered cards, including their mobile preview where relevant.

## Write the caption

The caption should:

- open with `[논문 한 편, 5장으로 읽기 · PAPER RADAR NNN]`;
- lead with an accessible question or practical tension;
- explain the problem, contribution, and result in plain Korean without exaggeration;
- state the review basis and limitations;
- include the complete paper title, authors, and canonical arXiv, DOI, or publisher link;
- clarify that the post is a paper-based summary.

Do not present legal or scientific certainty beyond the evidence. Linking to the canonical abstract or publisher page is preferred. Summarize in original wording.

## Choose hashtags

- Check current topic interest at preparation time because traffic changes.
- User preference (2026-09-10): starting with the next post, experiment with more discovery-oriented hashtags rather than a fixed five-tag set. Verify Instagram's current supported limit before expanding; if only five are allowed, improve the selection within that limit and explain this constraint. Do not bypass limits with comment stuffing.
- Build a broader candidate pool using current evidence of topic interest and relevant creator usage. Do not label tags as high-traffic without evidence or equate historical post counts with current reach.
- Balance broad discovery, Korean AI/topic, paper-specific technical, format tags such as `#논문요약`, and the brand archive tag `#한해원의관측소`. Include only tags genuinely relevant to the paper.
- Record the chosen set and, when Insights are available, compare non-follower reach, profile visits, and saves across posts; treat increased tag count as an experiment, not a promise of increased traffic.
- Do not fill the set with only high-volume generic tags; they are highly competitive and weak at describing the post.
- Keep the full paper name and rare project name in the caption when their hashtag traffic is negligible.

## Accumulate development value

For every selected paper, update both:

- `docs/연구_인사이트_원장.md` for the human-readable record.
- `data/research_insights.jsonl` for the append-only structured record, one JSON object per line.

Record verified paper insights, Radar/model application candidates, public code or data, license, maturity, an `adopt`/`experiment`/`watch`/`reject` decision, and one concrete next experiment. Use the same `item_id` with a new version when the assessment changes rather than silently rewriting history.

## Daily delivery and publishing

- Prepare the daily package at `07:40 Asia/Seoul`, targeting an `08:00–08:30` commute-time publication. Revisit timing after enough Instagram Insights data exists; compare morning with a late-morning test window rather than assuming a universal best time.
- Present the selected-paper rationale, verification basis, five cards, final caption, hashtags, and source link for review.
- A request to prepare recurring posts does not remove the final publication checkpoint. Immediately before Instagram's final public `Share` action, obtain explicit user confirmation. After confirmation, publish through the logged-in `@fromhaewon` Chrome session and report success or the exact blocker.
- Never claim the post succeeded unless the published post is visibly verified.
