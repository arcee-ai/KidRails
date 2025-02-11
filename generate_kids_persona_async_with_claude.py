import json
import asyncio
import sqlite3
import re
from typing import Dict, List
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm
from anthropic import AsyncAnthropic


ANTHROPIC_API_KEY = "sk-..." # Put your API key here!
MODEL_NAME = "claude-3-5-sonnet-20240620"
client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
rate_limiter = AsyncLimiter(5)


def make_prompt(age, developmental_stage, personality_traits, adjectives, interests):
    prompt_template = """You are an expert kids persona creator. Your task is to create a unique and detailed kids persona
based on the following input variables:

<age>{$AGE}</age>
<developmental_stage>{$DEVELOPMENTAL_STAGE}</developmental_stage>
<personality_traits>{$PERSONALITY_TRAITS}</personality_traits>
<adjectives>{$ADJECTIVES}</adjectives>
<interests>{$INTERESTS}</interests>

Follow these guidelines to create the persona:

1. Choose a name that fits the age and personality traits.
2. Incorporate the age, developmental stage, personality traits, adjectives, and interests
seamlessly into the description.
3. Provide context for how the child's interests and personality traits manifest in their daily
life.
4. Include details about the child's cognitive abilities, emotional needs, and social interactions.
5. Describe how the child's interests influence their behavior and relationships.
6. Ensure the persona is coherent, realistic, and age-appropriate.

Your output should be a paragraph of 100-150 words that vividly describes the child. The description
should flow naturally, integrating all the provided information without explicitly listing the input
variables.

Here's an example of a well-crafted persona:

<example>
Emma, a vivacious 8-year-old, embodies curiosity and creativity in her everyday adventures. In the
concrete operational stage, she eagerly applies logic to her passion for science experiments, often
transforming the kitchen into a makeshift laboratory. Emma's extroverted nature shines through her
enthusiasm for sharing discoveries with friends and family. Described as bright and imaginative, she
thrives on praise for her innovative ideas. While generally even-tempered, Emma occasionally
struggles with patience when experiments don't yield immediate results. Her interests extend beyond
science to include storytelling, where she weaves fantastical tales inspired by her backyard
explorations. Emma's persona is characterized by her inquisitive spirit, social nature, and the
delightful blend of scientific curiosity with creative expression.
</example>

Now, create a unique kids persona based on the provided input variables. Begin your response with
<persona> and end it with </persona>."""

    prompt_with_variables = prompt_template
    prompt_with_variables = prompt_with_variables.replace("{$AGE}", age)
    prompt_with_variables = prompt_with_variables.replace("{$DEVELOPMENTAL_STAGE}", interests)
    prompt_with_variables = prompt_with_variables.replace("{$PERSONALITY_TRAITS}", personality_traits)
    prompt_with_variables = prompt_with_variables.replace("{$ADJECTIVES}", adjectives)
    prompt_with_variables = prompt_with_variables.replace("{$INTERESTS}", interests)
    return prompt_with_variables



def pretty_print(message):
    pretty_message = '\n\n'.join('\n'.join(line.strip() for line in re.findall(r'.{1,100}(?:\s+|$)', paragraph.strip('\n'))) for paragraph in re.split(r'\n\n+', message))
    return pretty_message


def extract_persona(text):
    pattern = r"<persona>(.*?)</persona>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None 

async def generate_persona(age: str, developmental_stage: str, personality_traits: str, adjectives: str, interests: str) -> str:
    prompt_with_variables = make_prompt(age, developmental_stage, personality_traits, adjectives, interests)
    async with rate_limiter:
        message = await client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content":  prompt_with_variables
                },
            ],
        )
        print(message)
        message = message.content[0].text
        pretty_message = pretty_print(message)
        return extract_persona(pretty_message)
     

class Database:
    def __init__(self, db_name: str = 'results.db'):
        self.db_name = db_name
        self.create_table()

    def create_table(self):
        with sqlite3.connect(self.db_name) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    age TEXT, developmental_stage TEXT, personality_traits TEXT, adjectives TEXT, interests TEXT, persona TEXT,
                    UNIQUE(age, developmental_stage, personality_traits, adjectives, interests)
                )
            ''')

    def entry_exists(self, age: str, developmental_stage: str, personality_traits: str, adjectives: str, interests: str) -> bool:
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.execute('''
                SELECT 1 FROM results WHERE age = ? AND developmental_stage = ? AND personality_traits = ? AND adjectives = ? AND interests = ?
            ''', (age, developmental_stage, personality_traits, adjectives, interests))
            return cursor.fetchone() is not None

    def save_entry(self, age: str, developmental_stage: str, personality_traits: str, adjectives: str, interests: str, persona: str):
        with sqlite3.connect(self.db_name) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO results (age, developmental_stage, personality_traits, adjectives, interests, persona) VALUES (?, ?, ?, ?, ?, ?)
            ''', (age, developmental_stage, personality_traits, adjectives, interests, persona))

async def process_entries(db: Database, entries: List[Dict[str, str]]):
    async def process_single_entry(entry: Dict[str, str]):
        age, developmental_stage, personality_traits, adjectives, interests = str(entry['age']), ", ".join(entry['characteristics']), str(entry['personality_trait']), ", ".join(entry['adjectives']), ", ".join(entry['interest'])
        if not db.entry_exists(age, developmental_stage, personality_traits, adjectives, interests):
            persona = await generate_persona(age, developmental_stage, personality_traits, adjectives, interests)
            db.save_entry(age, developmental_stage, personality_traits, adjectives, interests, str(persona))

    tasks = [process_single_entry(entry) for entry in entries]
    for task in tqdm(asyncio.as_completed(tasks), total=len(entries), desc="Processing entries"):
        await task

async def main():
    db = Database()

    with open('kids_characteristics_pool.json', 'r') as f:
        dataset = json.load(f)
        print(dataset)

    await process_entries(db, dataset)

if __name__ == "__main__":
    asyncio.run(main())
