# Prompt Design — Humanitarian Crisis Classification

---

## Slide 1 — Zero-Shot Prompting

```
┌─────────────────────────────────────────────────────────────┐
│  SYSTEM                                                     │
│  ├─ Role: "expert humanitarian-crisis analyst"              │
│  ├─ 5 class definitions  (text cues + visual cues)         │
│  └─ Decision guide  (tiebreak rules for ambiguous cases)   │
├─────────────────────────────────────────────────────────────┤
│  USER                                                       │
│  ├─ 🖼  query image   ← injected by CrisisClassifier       │
│  ├─ Tweet: <tweet_text>                                     │
│  └─ "Category:"                                             │
├─────────────────────────────────────────────────────────────┤
│  MODEL OUTPUT                                               │
│  └─ not_humanitarian   (exact class name, nothing else)    │
└─────────────────────────────────────────────────────────────┘
```

| Thành phần | Nội dung |
|---|---|
| 5 class definitions | Mô tả ngắn text cues + visual cues cho từng class |
| Decision guide | 5 quy tắc tiebreak cho trường hợp ambiguous |
| Output constraint | Chỉ trả về đúng tên class, không giải thích |
| Pre-processing | ❌ Không — text giữ nguyên, image do `AutoProcessor` xử lý |

---

## Slide 2 — Few-Shot Prompting (5-shot · Multimodal Multi-turn)

```
┌──────────────────────────────────────────────────────────────┐
│  SYSTEM  (giống zero-shot)                                   │
├──────────────────────────────────────────────────────────────┤
│  USER     🖼 image_1  +  Tweet: <ex1>  +  "Category:"       │
│  ASSISTANT  affected_individuals                             │
├──────────────────────────────────────────────────────────────┤
│  USER     🖼 image_2  +  Tweet: <ex2>  +  "Category:"       │
│  ASSISTANT  infrastructure_and_utility_damage               │
├──────────────────────────────────────────────────────────────┤
│  USER     🖼 image_3  +  Tweet: <ex3>  +  "Category:"       │
│  ASSISTANT  not_humanitarian                                 │
├──────────────────────────────────────────────────────────────┤
│  USER     🖼 image_4  +  Tweet: <ex4>  +  "Category:"       │
│  ASSISTANT  other_relevant_information                       │
├──────────────────────────────────────────────────────────────┤
│  USER     🖼 image_5  +  Tweet: <ex5>  +  "Category:"       │
│  ASSISTANT  rescue_volunteering_or_donation_effort          │
├──────────────────────────────────────────────────────────────┤
│  USER  ← QUERY                                               │
│  🖼 query_image  ← injected by CrisisClassifier             │
│  Tweet: <query_tweet>  +  "Category:"                        │
├──────────────────────────────────────────────────────────────┤
│  MODEL OUTPUT                                                │
│  └─ <predicted_class>                                        │
└──────────────────────────────────────────────────────────────┘
```

**Example selection:** 1 sample per class, từ training split, seed=42.

| | Zero-shot | Few-shot (5-shot + img) |
|---|:---:|:---:|
| Số turn | 2 | 12 |
| Ảnh trong context | 1 | 6 |
| Inference time | ~0.5 s | ~0.9 s |
| F1-macro (2B model) | 0.275–0.364 | 0.313 |
