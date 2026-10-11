import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

with open(
    "training.txt",
    "r",
    encoding="utf-8"
) as f:
    general_text = f.read()


conversation_file = "ranger_conversation_training.txt"

if not os.path.exists(conversation_file):
    print()
    print("ERROR:")
    print("The file ranger_conversation_training.txt was not found.")
    print("Put it in the same folder as ranger_v2.py.")
    raise SystemExit


with open(
    conversation_file,
    "r",
    encoding="utf-8"
) as f:
    conversation_text = f.read()


print("Training text loaded.")
print("General/literature characters:", len(general_text))
print("Conversation characters:", len(conversation_text))


# ============================================================
# 2. CREATE ONE VOCABULARY
# ============================================================

all_text = general_text + conversation_text

chars = sorted(set(all_text))
vocab_size = len(chars)

print("Vocabulary size:", vocab_size)


stoi = {
    ch: i
    for i, ch in enumerate(chars)
}


itos = {
    i: ch
    for ch, i in stoi.items()
}


# ============================================================
# 3. ENCODER / DECODER
# ============================================================

def encode(text):
    return [
        stoi[c]
        for c in text
    ]


def decode(tokens):
    return "".join(
        itos[i]
        for i in tokens
    )


# ============================================================
# 4. TURN DATA INTO NUMBERS
# ============================================================

general_data = torch.tensor(
    encode(general_text),
    dtype=torch.long
)


conversation_data = torch.tensor(
    encode(conversation_text),
    dtype=torch.long
)


print("General data shape:", general_data.shape)
print("Conversation data shape:", conversation_data.shape)


# ============================================================
# 5. RANGER SETTINGS
# ============================================================

# Ranger is still deliberately small, but this is larger than
# the previous 119,804-parameter version.

block_size = 64

n_embd = 96

n_head = 4

n_layer = 3

batch_size = 16


# During normal training, this percentage of batches will come
# from the conversational corpus.
#
# This is the important change:
#
# We do NOT need to make the conversation file as large as the
# literature. We deliberately sample it more often.

conversation_batch_probability = 0.60


# Initial training length.

training_steps = 20000


# Generation settings.

temperature = 0.70

top_k = 20


# ============================================================
# 6. BATCH MAKER
# ============================================================

def make_batch_from_source(source):
    starts = torch.randint(
        0,
        len(source) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        source[
            i:i + block_size
        ]
        for i in starts
    ])

    y = torch.stack([
        source[
            i + 1:i + block_size + 1
        ]
        for i in starts
    ])

    return x, y


def get_training_batch():
    # Most batches come from the conversation corpus.
    # The remaining batches come from the original literature
    # and general training material.

    if torch.rand(1).item() < conversation_batch_probability:
        return make_batch_from_source(
            conversation_data
        )

    return make_batch_from_source(
        general_data
    )


# ============================================================
# 7. SELF-ATTENTION HEAD
# ============================================================

class Head(nn.Module):

    def __init__(
        self,
        head_size
    ):

        super().__init__()

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )


    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)

        q = self.query(x)

        weights = q @ k.transpose(
            -2,
            -1
        )

        weights = weights * (
            C ** -0.5
        )

        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(
            weights,
            dim=-1
        )

        v = self.value(x)

        out = weights @ v

        return out


# ============================================================
# 8. MULTI-HEAD ATTENTION
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_size
    ):

        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(
            n_embd,
            n_embd
        )


    def forward(self, x):

        out = torch.cat(
            [
                head(x)
                for head in self.heads
            ],
            dim=-1
        )

        return self.proj(out)


# ============================================================
# 9. FEED-FORWARD NETWORK
# ============================================================

class FeedForward(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                n_embd,
                4 * n_embd
            ),

            nn.ReLU(),

            nn.Linear(
                4 * n_embd,
                n_embd
            )
        )


    def forward(self, x):

        return self.net(x)


# ============================================================
# 10. TRANSFORMER BLOCK
# ============================================================

class Block(nn.Module):

    def __init__(self):

        super().__init__()

        head_size = (
            n_embd // n_head
        )

        self.sa = MultiHeadAttention(
            n_head,
            head_size
        )

        self.ffwd = FeedForward()

        self.ln1 = nn.LayerNorm(
            n_embd
        )

        self.ln2 = nn.LayerNorm(
            n_embd
        )


    def forward(self, x):

        x = x + self.sa(
            self.ln1(x)
        )

        x = x + self.ffwd(
            self.ln2(x)
        )

        return x


# ============================================================
# 11. RANGER MODEL
# ============================================================

class Ranger(nn.Module):

    def __init__(self):

        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.Sequential(
            *[
                Block()
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(
            n_embd
        )

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )


    def forward(self, idx):

        B, T = idx.shape

        tok_emb = self.token_embedding(
            idx
        )

        positions = torch.arange(
            T,
            device=idx.device
        )

        pos_emb = self.position_embedding(
            positions
        )

        x = tok_emb + pos_emb

        x = self.blocks(
            x
        )

        x = self.ln_f(
            x
        )

        logits = self.lm_head(
            x
        )

        return logits


# ============================================================
# 12. CREATE RANGER
# ============================================================

model = Ranger()

print()
print("Ranger model created.")

number_of_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Ranger parameters:",
    number_of_parameters
)


# ============================================================
# 13. MODEL CHECKPOINT
# ============================================================

checkpoint_file = "ranger_v3.pt"


def save_model():

    torch.save(
        {
            "model_state": model.state_dict(),
            "chars": chars,
            "block_size": block_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer
        },
        checkpoint_file
    )

    print("Ranger's weights saved.")


# ============================================================
# 14. LOAD EXISTING MODEL OR TRAIN FROM SCRATCH
# ============================================================

already_trained = False


if os.path.exists(
    checkpoint_file
):

    print()
    print("Saved Ranger model found.")

    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu"
    )

    same_vocabulary = (
        checkpoint.get("chars") == chars
    )

    same_architecture = (
        checkpoint.get("block_size") == block_size
        and checkpoint.get("n_embd") == n_embd
        and checkpoint.get("n_head") == n_head
        and checkpoint.get("n_layer") == n_layer
    )

    if same_vocabulary and same_architecture:

        model.load_state_dict(
            checkpoint["model_state"]
        )

        print(
            "Ranger's trained weights loaded."
        )

        already_trained = True

    else:

        print()
        print(
            "The existing ranger.pt was made with a different"
        )
        print(
            "vocabulary or model architecture."
        )
        print(
            "A new model will be trained."
        )


# ============================================================
# 15. INITIAL TRAINING
# ============================================================

if not already_trained:

    print()
    print(
        "Beginning balanced training..."
    )
    print()

    print(
        "Conversation batch probability:",
        conversation_batch_probability
    )

    print(
        "Training steps:",
        training_steps
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0003
    )

    for step in range(
        training_steps
    ):

        xb, yb = get_training_batch()

        logits = model(
            xb
        )

        loss = F.cross_entropy(
            logits.view(
                -1,
                vocab_size
            ),
            yb.view(
                -1
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        if step % 500 == 0:

            print(
                "Step:",
                step,
                "| Loss:",
                loss.item()
            )


    print()
    print(
        "Initial training complete."
    )

    save_model()


# ============================================================
# 16. TEXT GENERATION
# ============================================================

@torch.no_grad()
def generate(
    idx,
    max_new_tokens
):

    for _ in range(
        max_new_tokens
    ):

        idx_cond = idx[
            :,
            -block_size:
        ]

        logits = model(
            idx_cond
        )

        logits = logits[
            :,
            -1,
            :
        ]

        # Lower temperature makes Ranger less random.

        logits = logits / temperature

        # Keep only the most likely characters.

        values, indices = torch.topk(
            logits,
            min(
                top_k,
                logits.shape[-1]
            )
        )

        filtered_logits = torch.full_like(
            logits,
            float("-inf")
        )

        filtered_logits.scatter_(
            1,
            indices,
            values
        )

        probabilities = F.softmax(
            filtered_logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        idx = torch.cat(
            (
                idx,
                next_token
            ),
            dim=1
        )

    return idx


# ============================================================
# 17. SAVE CONVERSATION
# ============================================================

def save_conversation(
    conversation_text
):

    with open(
        "conversations.txt",
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            conversation_text
        )

        f.write(
            "\n\n"
        )


# ============================================================
# 18. START CONVERSATION
# ============================================================

print()
print("===================================")
print("Ranger is ready.")
print("Type 'exit' to leave.")
print("Type 'save' to save the conversation.")
print("===================================")
print()

conversation = ""


# ============================================================
# 19. CONVERSATION LOOP
# ============================================================

while True:

    user_text = input(
        "\nYou: "
    )


    if user_text.lower() == "exit":

        print()
        print(
            "Ranger is shutting down."
        )

        break


    if user_text.lower() == "save":

        save_conversation(
            conversation
        )

        print(
            "Conversation saved."
        )

        continue


    if user_text.strip() == "":
        continue


    # --------------------------------------------------------
    # Build the prompt in the same format Ranger saw during
    # training:
    #
    # Human: something
    # Ranger: something
    #
    # The final "Ranger:" tells the model that it is now
    # Ranger's turn to answer.
    # --------------------------------------------------------

    new_message = (
        "Human: "
        + user_text
        + "\n"
        "Ranger:"
    )


    # Keep a little room for the new prompt inside the
    # model's 64-character context window.

    context_limit = block_size - 16

    if len(conversation) > context_limit:
        conversation = conversation[-context_limit:]


    prompt_text = conversation + new_message


    # Make sure the prompt itself fits inside block_size.

    if len(prompt_text) > block_size:
        prompt_text = prompt_text[-block_size:]


    try:

        prompt_tokens = torch.tensor(
            [
                encode(
                    prompt_text
                )
            ],
            dtype=torch.long
        )

    except KeyError:

        print()
        print(
            "Ranger encountered a character "
            "that is not in its vocabulary."
        )

        continue


    prompt_length = prompt_tokens.shape[1]


    # Generate the answer after the Ranger: cue.

    output = generate(
        prompt_tokens,
        150
    )


    result = decode(
        output[0].tolist()
    )


    # Only keep text generated AFTER the prompt.

    answer = result[
        prompt_length:
    ]


    # Stop when Ranger starts generating the next Human turn.

    if "Human:" in answer:

        answer = answer.split(
            "Human:"
        )[0]


    # Occasionally the model may repeat its own label.

    if answer.startswith("Ranger:"):

        answer = answer[
            len("Ranger:"):
        ]


    answer = answer.strip()


    # If the model somehow produced nothing, try one more
    # generation with the same prompt instead of silently
    # giving a blank response.

    if answer == "":

        output = generate(
            prompt_tokens,
            100
        )

        result = decode(
            output[0].tolist()
        )

        answer = result[
            prompt_length:
        ]

        if "Human:" in answer:

            answer = answer.split(
                "Human:"
            )[0]

        if answer.startswith("Ranger:"):

            answer = answer[
                len("Ranger:"):
            ]

        answer = answer.strip()


    if answer == "":

        answer = (
            "I don't have a good response to that yet."
        )


    print(
        "Ranger:",
        answer
    )


    # Save the exchange as conversation context.

    exchange = (
        "Human: "
        + user_text
        + "\n"
        "Ranger: "
        + answer
        + "\n"
    )

    conversation += exchange


    # Keep the live context small enough for the model.

    if len(conversation) > block_size:

        conversation = conversation[
            -block_size:
        ]


    # Save the exchange to the conversation log.
    #
    # IMPORTANT:
    # We are NOT training the model's weights on every
    # conversation yet. This keeps the model stable while
    # we work on the basic conversational behavior.

    save_conversation(
        exchange
    )


# ============================================================
# END
# ============================================================
