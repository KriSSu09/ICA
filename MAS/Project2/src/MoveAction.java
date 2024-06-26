class MoveAction extends Action {
    private final Position position;

    public MoveAction(Position position) {
        this.position = position;
    }

    @Override
    public State execute(Agent a, State s) {
        var agent = (TagAgent) a;
        agent.getAgentState().setPosition(position);
        if (!agent.getAgentState().isIt()) {
            agent.getAgentState().incrementScore();
        }
        var state = (TagState) s;
        state.setAgentById(agent);
        return state;
    }

    @Override
    public String toString() {
        return "MoveAction{" +
                "position=" + position +
                '}';
    }
}