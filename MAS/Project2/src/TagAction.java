class TagAction extends Action {
    private final Position agentToTagPosition;

    public TagAction(Position agentToTagPosition) {
        this.agentToTagPosition = agentToTagPosition;
    }

    @Override
    public State execute(Agent a, State s) {
        var state = (TagState) s;
        TagAgent agentToTag = state.getAgentByPosition(agentToTagPosition);
        AgentState agentToTagState = agentToTag.getAgentState();
        agentToTagState.setIt(true);
        agentToTagState.setCanTag(3);
        agentToTagState.decrementNonItScore();
        var agent = (TagAgent) a;
        agent.getAgentState().incrementScore();
        agent.getAgentState().setIt(false);
        state.setAgentById(agent);
        state.setAgentById(agentToTag);
        return state;
    }

    @Override
    public String toString() {
        return "TagAction{" +
                "agentToTagPosition=" + agentToTagPosition +
                '}';
    }
}