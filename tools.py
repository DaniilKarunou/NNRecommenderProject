import random

def generate_negative_interactions(interactions_df, n_neg_per_pos=5):

    # Get a set of all (user_id, item_id) pairs in the interactions_df
    existing_pairs = set(zip(interactions_df['user_id'], interactions_df['item_id']))

    # Get a list of all user_ids
    users_list = interactions_df['user_id'].unique()

    # Get a list of all item_ids
    items_list = interactions_df['item_id'].unique()

    # Generate negative interactions by randomly sampling pairs not in the existing_pairs set
    negative_interactions = []
    while len(negative_interactions) < n_neg_per_pos * len(interactions_df):
        user_id = random.choice(users_list)
        item_id = random.choice(items_list)
        if (user_id, item_id) not in existing_pairs:
            negative_interactions.append((user_id, item_id, 0))

    return negative_interactions