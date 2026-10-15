import pandas as pd
from config import MUSIC_CSV, EMOTION_TO_MOOD


def label_mood(row):
    v, e = row['valence'], row['energy']
    if v >= 0.5 and e >= 0.5:
        return 'Happy'
    elif v >= 0.5 and e < 0.5:
        return 'Calm'
    elif v < 0.5 and e >= 0.5:
        return 'Angry'
    else:
        return 'Sad'


def load_music():
    music_df = pd.read_csv(MUSIC_CSV)
    music_df.columns = [c.strip().lower() for c in music_df.columns]
    print(f"Music loaded: {music_df.shape}")
    print("Columns:", music_df.columns.tolist())

    if 'mood' not in music_df.columns:
        music_df['mood'] = music_df.apply(label_mood, axis=1)

    print("\n Mood distribution:")
    print(music_df['mood'].value_counts())

    return music_df


def recommend_songs(music_df, emotion, top_k=5):
    mood = EMOTION_TO_MOOD.get(emotion)
    if mood is None:
        return pd.DataFrame()

    subset = music_df[music_df['mood'] == mood]

    if subset.empty:
        print(f"No songs for mood '{mood}' — showing top popular overall.")
        subset = music_df

    cols = [c for c in ['artist_name', 'track_name', 'popularity',
                        'valence', 'energy', 'mood'] if c in subset.columns]
    return subset.sort_values('popularity', ascending=False).head(top_k)[cols].reset_index(drop=True)


if __name__ == "__main__":
    df = load_music()
    for emo in ['Sad', 'Happy', 'Angry', 'Neutral']:
        print(f"\n {emo} → mood '{EMOTION_TO_MOOD[emo]}':")
        print(recommend_songs(df, emo, top_k=5))