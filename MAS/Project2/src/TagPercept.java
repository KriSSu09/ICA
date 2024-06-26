import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class TagPercept extends Percept {
    private final Position closestAgentPosition;
    private final int[] canMoveDirection = {1, 1, 1, 1};

    public TagPercept(TagState state, Agent a) {
        super(state, a);
        var agent = (TagAgent) a;
        var agentPositions = Simulation.blackboard.readAllPositions();
        var otherAgentPositions = agentPositions.entrySet().stream()
                .filter(e -> !e.getKey().equals(agent.getId()))
                .map(Map.Entry::getValue)
                .collect(Collectors.toSet());
        this.closestAgentPosition = findNearestAgentPosition(agent, otherAgentPositions);
        int[] dx = {0, 1, 0, -1};
        int[] dy = {-1, 0, 1, 0};
        for (int i = 0; i < 4; i++) {
            int finalI = i;
            if (agent.getAgentState().getPosition().x() + dx[i] < 0 ||
                    agent.getAgentState().getPosition().x() + dx[i] >= state.getWidth() ||
                    agent.getAgentState().getPosition().y() + dy[i] < 0 ||
                    agent.getAgentState().getPosition().y() + dy[i] >= state.getHeight() ||
                    state.getObstacles().contains(new Position(agent.getAgentState().getPosition().x() + dx[i], agent.getAgentState().getPosition().y() + dy[i])) ||
                    otherAgentPositions.stream().anyMatch(p -> p.equals(new Position(agent.getAgentState().getPosition().x() + dx[finalI], agent.getAgentState().getPosition().y() + dy[finalI])))
            ) {
                canMoveDirection[i] = 0;
            }
        }
    }

    public Position getClosestAgentPosition() {
        return closestAgentPosition;
    }

    public int[] getCanMoveDirection() {
        return canMoveDirection;
    }

    @Override
    public String toString() {
        return "TagPercept{" +
                "closestAgentPosition=" + closestAgentPosition +
                ", canMoveDirection=" + java.util.Arrays.toString(canMoveDirection) +
                '}';
    }

    private Position findNearestAgentPosition(TagAgent agent, Set<Position> otherAgentPositions) {
        Position currentPosition = agent.getAgentState().getPosition();
        double minDistance = Double.MAX_VALUE;
        Position nearestPosition = null;

        for (Position otherPosition : otherAgentPositions) {
            double distance = currentPosition.distanceTo(otherPosition);
            if (distance < minDistance) {
                minDistance = distance;
                nearestPosition = otherPosition;
            }

        }

        return nearestPosition;
    }
}
