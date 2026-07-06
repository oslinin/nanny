# Sources & licenses for the `child-guidance` skill

`SKILL.md` contains **original plain-language summaries** of mainstream,
evidence-based child-health guidance. It does **not** reproduce any source's
text verbatim. Facts and general guidance are not themselves copyrightable;
this file records where each summarized idea comes from so answers can cite a
reputable source and a reader can go read the original.

If you extend `SKILL.md` with **direct quotations**, only quote from the CC BY
or public-domain sources below (with attribution) — do not paste text from a
source whose license you have not confirmed permits it.

## Sources

| Source | Used for | License / status | Link |
|---|---|---|---|
| UNICEF — *The Art of Parenting* (training guide) & UNICEF Parenting | Responsive caregiving, feeding on cue, hydration signals — the backbone reference the parent asked to use | ⚠️ **License not yet confirmed** — UNICEF publications are often CC BY-NC-SA 3.0 IGO, but some are traditional ©. Only summarized/cited here, never reproduced, so this is safe regardless; **confirm before quoting**. | https://www.unicef.org/belize/documents/art-parenting-training-guide-and-summary-parents-and-caregivers · https://www.unicef.org/parenting/ |
| CDC — infant nutrition, sleep, and "Learn the Signs. Act Early." | Feeding, sleep amounts, developmental milestone ranges | Public domain (US Government work) | https://www.cdc.gov/nutrition/infantandtoddlernutrition/ · https://www.cdc.gov/ncbddd/actearly/ |
| AAP / HealthyChildren.org | Bottle-feeding, stool/diaper norms, safe sleep, when to call a pediatrician | © AAP — summarized/cited only, not reproduced | https://www.healthychildren.org |
| WHO — infant & young child feeding | ~6-month complementary-feeding window | © WHO (often CC BY-NC-SA 3.0 IGO) — summarized/cited only | https://www.who.int/health-topics/breastfeeding |
| USDA / FNS — *Feeding Infants* (CACFP) | Introducing solids, iron-rich first foods | Public domain (US Government work) | https://www.fns.usda.gov/tn/feeding-infants-child-and-adult-care-food-program |
| OpenStax — *Lifespan Development* | Early development, milestone ranges | **CC BY 4.0** (quotable with attribution) | https://openstax.org/books/lifespan-development |

## Build-time TODO

- [ ] Confirm the exact license printed on the copyright page of UNICEF's *The
      Art of Parenting* PDF before shipping any build that **quotes** it. (It
      could not be fetched from the development sandbox — UNICEF's site returned
      HTTP 403 there.) The current `SKILL.md` only summarizes and cites it, which
      needs no such permission, but a future verbatim-quote change would.
