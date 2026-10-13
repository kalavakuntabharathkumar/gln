from pathlib import Path
import random
import pandas as pd
from gradelens.data import initialize_db, save_grades

random.seed(7)
Path("data").mkdir(exist_ok=True)
topics = [
    ("photosynthesis", "Plants use sunlight, water and carbon dioxide to make glucose and oxygen."),
    ("gravity", "Gravity is the attractive force between objects with mass."),
    ("evaporation", "Evaporation is the change of a liquid into gas at its surface."),
    ("mitosis", "Mitosis is cell division that produces two genetically similar daughter cells."),
    ("ecosystem", "An ecosystem includes organisms and their physical environment interacting together.")
]
rows = []
for i in range(320):
    topic, definition = random.choice(topics)
    score = random.choice([0, 1, 2, 3])
    answers = [f"I do not know what {topic} means.", f"{topic} is a science thing.",
               definition.split(".")[0] + ".", definition]
    rows.append({"student_id": f"STU-{i+1:03}", "question_id": f"Q-{topic.upper()}",
        "answer_text": answers[score], "teacher_score": score,
        "predicted_score": max(0, min(3, score + random.choice([-1, 0, 0, 0, 1]))),
        "confidence": round(random.uniform(.62, .99), 2)})
df = pd.DataFrame(rows)
df.to_csv("data/demo_answers.csv", index=False)
initialize_db()
save_grades(df)
print("Created data/demo_answers.csv and data/gradelens.db")
