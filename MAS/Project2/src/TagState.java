import java.util.*;

public class TagState extends State {
    private int width;
    private int height;
    private Set<Position> obstacles;
    private Map<Integer, Agent> agents;
    private int iteration;

    public static TagState getInitState(int width, int height, int numObstacles, int numAgents) {
        TagState state = new TagState();
        state.width = width;
        state.height = height;
        state.iteration = 0;
        state.obstacles = new HashSet<>();
        while (state.obstacles.size() < numObstacles) {
            state.obstacles.add(new Position((int) (Math.random() * width), (int) (Math.random() * height)));
        }
        state.agents = new HashMap<>();
        for (int i = 0; i < numAgents; i++) {
            Position position = new Position((int) (Math.random() * width), (int) (Math.random() * height));
            state.agents.put(i, new TagAgent(new AgentState(position, false), i));
            Simulation.blackboard.writePosition(state.agents.get(i));
        }
        ((TagAgent) state.agents.get((int) (Math.random() * numAgents))).getAgentState().setIt(true);
        return state;
    }

    public void setAgentById(Agent a) {
        var agent = (TagAgent) a;
        agents.put(agent.getId(), agent);
    }

    public Set<Position> getObstacles() {
        return obstacles;
    }

    public int getIteration() {
        return iteration;
    }

    public void incrementIteration() {
        iteration++;
    }

    public Set<Agent> getAgents() {
        return new HashSet<>(agents.values());
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public TagAgent getAgentByPosition(Position position) {
        for (var a : agents.values()) {
            var agent = (TagAgent) a;
            if (agent.getAgentState().getPosition().equals(position)) {
                return agent;
            }
        }
        return null;
    }

    @Override
    public void display() {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                final Position position = new Position(x, y);
                if (obstacles.contains(position)) {
                    System.out.print("X ");
                } else {
                    boolean printed = false;
                    for (var a : agents.values()) {
                        var agent = (TagAgent) a;
                        if (agent.getAgentState().getPosition().equals(position)) {
                            System.out.print(agent.getAgentState().isIt() ? "I " : "N ");
                            printed = true;
                            break;
                        }
                    }
                    if (!printed) {
                        System.out.print(". ");
                    }
                }
            }
            System.out.println();
        }
        System.out.println();
    }
}
