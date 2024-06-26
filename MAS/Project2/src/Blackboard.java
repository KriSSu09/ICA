import java.util.HashMap;
import java.util.Map;

public class Blackboard {
    private final Map<Integer, Position> positions;

    public Blackboard() {
        this.positions = new HashMap<>();
    }

    public void writePosition(Agent a) {
        TagAgent agent = (TagAgent) a;
        positions.put(agent.getId(), agent.getAgentState().getPosition());
    }

    public Map<Integer, Position> readAllPositions() {
        return new HashMap<>(positions);
    }
}
