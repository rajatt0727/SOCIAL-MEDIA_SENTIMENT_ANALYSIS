# src/synthetic_data_generation.py
import pandas as pd
import random
import faker

def generate_synthetic_data(num_posts=1000):
    fake = faker.Faker()
    data = []
    for _ in range(num_posts):
        post_id = fake.uuid4()
        user = fake.user_name()
        date = fake.date_time_this_year()
        content = fake.text(max_nb_chars=random.randint(50, 200))
        data.append([post_id, user, date, content])
    df = pd.DataFrame(data, columns=['PostID', 'User', 'Date', 'Content'])
    return df

# Generate synthetic data
synthetic_data = generate_synthetic_data()
synthetic_data.to_csv('data/synthetic_tweets.csv', index=False)
print("Synthetic data generated successfully.")
