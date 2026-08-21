"""
generate_data.py
-----------------
Creates realistic synthetic customer feedback by COMPOSING reviews from
word/phrase banks (subject + opinion + reason + closer), rather than
picking whole fixed sentences. This produces thousands of genuinely
distinct phrasings, so the classification task requires real
generalization instead of memorizing a handful of exact templates.
"""

import random
import pandas as pd

random.seed(42)

subjects = [
    "The product", "This item", "The service", "My order", "This purchase",
    "The delivery", "The support team", "The app", "The device", "This brand",
    "The package", "The website", "Customer support", "The quality", "The material",
]

positive_openers = [
    "exceeded my expectations", "worked perfectly", "was excellent",
    "impressed me a lot", "was exactly what I needed", "was fantastic",
    "performed really well", "was a great experience", "was top notch",
    "was outstanding", "made me very happy", "was better than expected",
    "was a pleasant surprise", "worked flawlessly", "was superb",
]
negative_openers = [
    "was a disappointment", "did not work as expected", "was frustrating",
    "was a letdown", "fell apart quickly", "was faulty", "was a waste of money",
    "was poor quality", "never worked properly", "was a bad experience",
    "arrived damaged", "was defective", "was terrible", "stopped working fast",
    "was completely unsatisfactory",
]
neutral_openers = [
    "was okay", "did the basic job", "was about average", "was fine, nothing more",
    "worked as expected, nothing special", "was acceptable", "was reasonable",
    "was neither good nor bad", "met basic expectations", "was standard",
    "was decent enough", "was so-so", "was ordinary", "worked most of the time",
]

positive_reasons = [
    "the quality was great", "delivery was fast", "the staff were helpful",
    "it was easy to use", "the price was fair", "everything worked smoothly",
    "support responded quickly", "the packaging was excellent",
    "it looked exactly as described", "the setup was simple",
]
negative_reasons = [
    "the quality was poor", "delivery took forever", "the staff were rude",
    "it was hard to use", "the price was not worth it", "nothing worked smoothly",
    "support never responded", "the packaging was damaged",
    "it looked nothing like described", "the setup was a nightmare",
]
neutral_reasons = [
    "the quality was average", "delivery was on time", "the staff were polite enough",
    "it was somewhat easy to use", "the price was fair for what it is",
    "some things worked, some did not", "support responded eventually",
    "the packaging was fine", "it mostly matched the description",
    "the setup took a normal amount of time",
]

closers = [
    "", "", "Would consider buying again.", "Just my honest review.",
    "That is my honest take.", "Sharing this for other buyers.",
    "Hope this review helps someone.", "Will update if anything changes.",
    "Overall that sums up my experience.", "",
]

fillers = [
    "", "Honestly, ", "To be fair, ", "Overall, ", "In my experience, ",
    "So far, ", "Personally, ", "All things considered, ", "Just my two cents, ",
    "After using it a while, ",
]

# ---- Hard/ambiguous mixed-sentiment sentence builder ----
def build_mixed(pos_part, neg_part, label_first=True):
    """Combine a positive clause and a negative clause -- label depends on which dominates."""
    subj = random.choice(subjects)
    if label_first:
        return f"{subj} {pos_part}, but {neg_part}."
    return f"{subj} {neg_part}, but {pos_part}."


def build_simple(sentiment):
    subj = random.choice(subjects)
    filler = random.choice(fillers)
    closer = random.choice(closers)
    if sentiment == "Positive":
        opener = random.choice(positive_openers)
        reason = random.choice(positive_reasons)
    elif sentiment == "Negative":
        opener = random.choice(negative_openers)
        reason = random.choice(negative_reasons)
    else:
        opener = random.choice(neutral_openers)
        reason = random.choice(neutral_reasons)

    structure = random.choice([1, 2, 3])
    if structure == 1:
        text = f"{filler}{subj} {opener}."
    elif structure == 2:
        text = f"{filler}{subj} {opener} because {reason}."
    else:
        text = f"{filler}{subj} {opener}. {reason.capitalize()}."

    if closer:
        text += " " + closer
    return text.strip()


rows = []

# simple, clearly-signaled reviews (majority of data)
for _ in range(620):
    rows.append((build_simple("Positive"), "Positive"))
for _ in range(580):
    rows.append((build_simple("Negative"), "Negative"))
for _ in range(520):
    rows.append((build_simple("Neutral"), "Neutral"))

# harder mixed-sentiment reviews -- label follows the DOMINANT clause
# (the clause after "but" is what the reviewer emphasizes/remembers most)
for _ in range(320):
    pos = random.choice(positive_reasons)
    neg = random.choice(negative_reasons)
    text = build_mixed(f"was okay since {pos}", neg, label_first=True)
    rows.append((text, "Negative"))

for _ in range(320):
    neg = random.choice(negative_reasons)
    pos = random.choice(positive_reasons)
    text = build_mixed(f"had issues since {neg}", pos, label_first=False)
    rows.append((text, "Positive"))

for _ in range(280):
    pos = random.choice(positive_reasons)
    neg = random.choice(negative_reasons)
    subj = random.choice(subjects)
    text = f"{subj} had some good points like {pos}, but also some downsides like {neg}. Mixed feelings overall."
    rows.append((text, "Neutral"))

random.shuffle(rows)

# Realistic label noise: ~3% of rows get a randomly reassigned label,
# mimicking real annotators occasionally disagreeing / mislabeling data.
NOISE_RATE = 0.03
all_labels = ["Positive", "Negative", "Neutral"]
noisy_rows = []
for text, label in rows:
    if random.random() < NOISE_RATE:
        other_labels = [l for l in all_labels if l != label]
        label = random.choice(other_labels)
    noisy_rows.append((text, label))
rows = noisy_rows

df = pd.DataFrame(rows, columns=["review_text", "sentiment"])
df = df.drop_duplicates(subset="review_text").reset_index(drop=True)
df.insert(0, "review_id", [f"R{2000+i}" for i in range(len(df))])

df.to_csv("/home/claude/sentiment_project/customer_feedback.csv", index=False)
print("Generated", len(df), "reviews (after de-dup)")
print(df["sentiment"].value_counts())
print(df.sample(8, random_state=1).to_string())
