public class TagAgent extends Agent {
    private final Integer id;
    private final AgentState agentState;
    private Position closestAgentPosition;
    private int[] canMoveDirection;

    public TagAgent(AgentState state, Integer id) {
        this.agentState = state;
        this.id = id;
    }

    public AgentState getAgentState() {
        return agentState;
    }

    public Integer getId() {
        return id;
    }

    @Override
    public void see(Percept p) {
        TagPercept tagPercept = (TagPercept) p;
        closestAgentPosition = tagPercept.getClosestAgentPosition();
        canMoveDirection = tagPercept.getCanMoveDirection();
    }

    @Override
    public Action selectAction() {
        if (agentState.isIt()) {
            if (agentState.canTag()) {
                if (closestAgentPosition.distanceTo(agentState.getPosition()) < 2) {
                    return new TagAction(closestAgentPosition);
                }
            } else {
                agentState.decrementCanTag();
            }
            return new MoveAction(getPositionToMoveTowards(agentState));
        } else {
            return new MoveAction(getPositionToMoveAway(agentState));
        }
    }

    private Position getPositionToMoveAway(AgentState agentState) {
        int[] dx = {0, 1, 0, -1};
        int[] dy = {-1, 0, 1, 0};
        Position positionToMove = null;
        for (int i = 0; i < 4; i++) {
            Position newPosition = new Position(agentState.getPosition().x() + dx[i], agentState.getPosition().y() + dy[i]);
            if (canMoveDirection[i] == 0)
                continue;
            if (positionToMove == null || newPosition.distanceTo(closestAgentPosition) > positionToMove.distanceTo(closestAgentPosition))
                positionToMove = newPosition;
        }
        return positionToMove;
    }

    private Position getPositionToMoveTowards(AgentState agentState) {
        int[] dx = {0, 1, 0, -1};
        int[] dy = {-1, 0, 1, 0};
        Position positionToMove = null;
        for (int i = 0; i < 4; i++) {
            Position newPosition = new Position(agentState.getPosition().x() + dx[i], agentState.getPosition().y() + dy[i]);
            if (canMoveDirection[i] == 0)
                continue;
            if (positionToMove == null || newPosition.distanceTo(closestAgentPosition) < positionToMove.distanceTo(closestAgentPosition))
                positionToMove = newPosition;
        }
        return positionToMove;
    }
}
