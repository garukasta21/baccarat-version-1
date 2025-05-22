import streamlit as st
import torch

class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.net(x.float())

model = DQN(10, 4)
model.load_state_dict(torch.load("dqn_baccarat_model.pth", map_location=torch.device('cpu')))
model.eval()

def suggest_bet(past_results):
    state = torch.tensor(past_results).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state)
    action = torch.argmax(q_values).item()
    return {0: 'Banker', 1: 'Player', 2: 'Tie', 3: 'Sit Out'}[action]

st.title("🎰 Baccarat Assistant")
st.markdown("Enter last 10 outcomes (0 = Banker, 1 = Player, 2 = Tie)")

input_text = st.text_input("Example: 0 1 0 1 2 0 1 1 0 2")

if st.button("🎯 Suggest Bet"):
    try:
        values = list(map(int, input_text.strip().split()))
        if len(values) != 10 or any(v not in [0, 1, 2] for v in values):
            st.error("Enter exactly 10 numbers: 0, 1, or 2.")
        else:
            prediction = suggest_bet(values)
            st.success(f"✅ Suggested Bet: **{prediction}**")
    except Exception as e:
        st.error(f"Error: {e}")
