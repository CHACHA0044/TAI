import os
import sys

# Change default encoding to UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from inference.engine import InferenceEngine

cases = [
    "Drinking bleach can cure COVID-19 within 24 hours according to scientists.",
    "The Earth revolves around the Sun and completes one orbit approximately every 365 days.",
    "Vaccines can cause severe side effects in a large number of people.",
    "All politicians are corrupt and only work for their own benefit.",
    "In conclusion, it is evident that technological advancements have significantly impacted modern society in numerous ways, shaping the future of humanity.",
    "Artificial intelligence is transforming industries by automating repetitive tasks and enabling data-driven decision making across sectors.",
    "I tried that new cafe yesterday and honestly the coffee was great but the service was kinda slow.",
    "The moon landing was staged in a Hollywood studio by the US government.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "Climate change is primarily caused by human activities such as burning fossil fuels and deforestation.",
    "Breaking news: Scientists confirm that eating chocolate daily can double your lifespan.",
    "Recent studies suggest that drinking silver nanoparticles can enhance immune response and prevent all viral infections."
]

engine = InferenceEngine()
for i, c in enumerate(cases, 1):
    res = engine.analyze(c)
    print(f"[{i}] {c[:50]}...")
    print(f"  Truth: {res['truth_score']} | AI: {res['ai_generated_score']} | Bias: {res['bias_score']} | Cred: {res['credibility_score']}")
    print("-" * 50)
