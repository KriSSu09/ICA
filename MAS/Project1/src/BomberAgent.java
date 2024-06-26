import jade.core.Agent;
import jade.core.behaviours.CyclicBehaviour;
import jade.core.behaviours.TickerBehaviour;
import jade.lang.acl.ACLMessage;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class BomberAgent extends Agent {
    private int agent_x;
    private int agent_y;
    private int last_x = -1;
    private int last_y = -1;
    private Grid grid;
    private boolean isAlive = true;
    private boolean isSmart;
    private String nickname;
    private Map<String, List<Integer>> otherAgentsPositions = new ConcurrentHashMap<>();
    private Map<StateActionPair, Double> qTable = new HashMap<>();
    private Random random = new Random();
    private final double alpha = 0.1;  // Learning rate
    private final double gamma = 0.9;  // Discount factor
    private final double epsilon = 0.2;  // Exploration rate
    private int score = 0;
    private int[] lastState;
    private int lastAction;
    private int moveCount = 0;

    // Send an ACLMessage to all other agents with the agent's position
    private void sendLocationUpdate() {
        AgentComms.broadcastToBomberAgents(this, nickname + ":" + agent_x + "," + agent_y);
    }

    private void handleBroadcastMessage(String content) {
        // The content format is "AgentName:x,y"
        String[] parts = content.split(":");
        String[] coords = parts[1].split(",");
        int x = Integer.parseInt(coords[0]);
        int y = Integer.parseInt(coords[1]);
        otherAgentsPositions.put(parts[0], Arrays.asList(x, y));
    }

    // Perception method to detect the first non-empty block in each direction
    private int[] perceiveEnvironment() {
        int[] state = new int[14]; // 4 directions * 2 data points per direction (block type, empty count) + 3 other agents * 2 coordinates
        int index = 0;

        // Directions: up, right, down, left
        int[] dx = {-1, 0, 1, 0};
        int[] dy = {0, 1, 0, -1};
        for (int direction = 0; direction < dx.length; direction++) {
            int nx = agent_x;
            int ny = agent_y;
            int emptyCount = 0;
            while (true) {
                nx += dx[direction];
                ny += dy[direction];
                if (nx < 0 || nx >= grid.getHeight() || ny < 0 || ny >= grid.getWidth()) {
                    state[index++] = -1; // Use -1 to indicate out of bounds
                    state[index++] = emptyCount;
                    break;
                }
                char cell = grid.getCell(nx, ny);
                if (cell != Grid.EMPTY) {
                    // Determine the type of the first non-empty cell encountered
                    if (cell == Grid.INDESTRUCTIBLE_WALL) {
                        state[index++] = 1; // Wall
                    } else if (cell == Grid.DESTRUCTIBLE_BLOCK) {
                        state[index++] = 2; // Destructible block
                    } else if (cell == Grid.AGENT) {
                        state[index++] = 3; // Another agent
                    } else if (cell == Grid.BOMB) {
                        state[index++] = 4;
                    } else {
                        state[index++] = 0; // Other types if necessary
                    }
                    state[index++] = emptyCount;
                    break;
                }
                emptyCount++;
            }
        }
        for(List<Integer> coords : otherAgentsPositions.values()) {
            state[index++] = coords.get(0);
            state[index++] = coords.get(1);
        }
        return state;
    }

    public int getMoveCount() {
        return moveCount;
    }

    public void increaseMoveCount() {
        moveCount++;
    }

    // Method to handle agent death
    public void die() {
        isAlive = false;
    }

    // Method to get the agent's x position
    public int getX() {
        return agent_x;
    }

    // Method to get the agent's y position
    public int getY() {
        return agent_y;
    }

    public int getScore() {
        return score;
    }

    public void updateScore(int score) {
        this.score += score;
        if (Objects.isNull(lastState)) {
            lastState = perceiveEnvironment();  // Convert perceptions to state
            lastAction = chooseAction(lastState);
        }
        int[] newState = perceiveEnvironment();  // Get new state after action
        updateQTable(lastState, lastAction, newState, score);  // Update Q-Table
        lastState = newState;
    }

    public boolean isAlive() {
        return isAlive;
    }

    public void performAction(int action) {
        int new_x = agent_x;
        int new_y = agent_y;
        switch (action) {
            case 0:
                new_x--;
                break;
            case 1:
                new_y++;
                break;
            case 2:
                new_x++;
                break;
            case 3:
                new_y--;
                break;
            case 4:
                // Place a bomb
                if (last_x > 0 && last_y > 0 && (agent_x != last_x || agent_y != last_y)) {
                    grid.placeBomb(agent_x, agent_y, 5, last_x, last_y, this);
                    agent_x = last_x;
                    agent_y = last_y;
                    break;
                }
        }
        // Check if the new position is valid
        if (grid.isValidMove(new_x, new_y)) {
            grid.updatePosition(agent_x, agent_y, new_x, new_y, Grid.AGENT);
            last_x = agent_x;
            last_y = agent_y;
            agent_x = new_x;
            agent_y = new_y;
            sendLocationUpdate();
        }
    }

    private int chooseAction(int[] state) {
        if (random.nextDouble() < epsilon) {  // Explore
            return random.nextInt(5);  // 4 directions + 1 bomb
        } else {  // Exploit
            return bestAction(state);
        }
    }

    private int bestAction(int[] state) {
        double bestValue = Double.NEGATIVE_INFINITY;
        int bestAction = 0;
        for (int a = 0; a < 5; a++) {
            StateActionPair sap = new StateActionPair(state, a);
            double value = qTable.getOrDefault(sap, 0.0);
            if (value > bestValue) {
                bestValue = value;
                bestAction = a;
            }
        }
        return bestAction;
    }

    private void updateQTable(int[] oldState, int action, int[] newState, double reward) {
        StateActionPair oldSAP = new StateActionPair(oldState, action);
        double oldQ = qTable.getOrDefault(oldSAP, 0.0);
        int bestNextAction = bestAction(newState);
        StateActionPair newSAP = new StateActionPair(newState, bestNextAction);
        double maxNewQ = qTable.getOrDefault(newSAP, 0.0);
        double newQ = oldQ + alpha * (reward + gamma * maxNewQ - oldQ);
        qTable.put(oldSAP, newQ);
    }

    public void saveQTable(String filepath) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filepath))) {
            oos.writeObject(qTable);
        } catch (IOException e) {
            System.err.println("Error saving Q-Table: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public void loadQTable(String filepath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filepath))) {
            qTable = (Map<StateActionPair, Double>) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error loading Q-Table: " + e.getMessage());
            e.printStackTrace();
        }
    }

    protected void setup() {
        // Receive initial position as arguments
        Object[] args = getArguments();
        if (args != null && args.length == 4) {
            agent_x = Integer.parseInt(args[0].toString());
            agent_y = Integer.parseInt(args[1].toString());
            grid = (Grid) args[2];
            isSmart = Boolean.parseBoolean(args[3].toString());
            nickname = getLocalName();
            grid.registerAgent(this);
        } else {
            throw new IllegalArgumentException("Arguments required: initial x and y positions, grid reference, smart");
        }
        if (isSmart) {
            loadQTable("qtable.ser");
        }
        addBehaviour(new TickerBehaviour(this, 10) {
            @Override
            public void onTick() {
                // Check if agent is alive
                if (!isAlive) {
                    return;
                }
                int action;
                if (!isSmart) {
                    // Perform random actions
                    action = (int) (Math.random() * 5);
                } else {
                    lastState = perceiveEnvironment();  // Convert perceptions to state
                    action = chooseAction(lastState);  // Decide on action
                    lastAction = action;
                }
                performAction(action);
            }
        });
        addBehaviour(new CyclicBehaviour(this) {
            public void action() {
                ACLMessage msg = receive();
                if (msg != null) {
                    handleBroadcastMessage(msg.getContent());
                }
                block();
            }
        });

    }

    private class StateActionPair implements Serializable {
        int[] state;
        int action;

        public StateActionPair(int[] state, int action) {
            this.state = state.clone();
            this.action = action;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            StateActionPair that = (StateActionPair) o;
            return action == that.action && Arrays.equals(state, that.state);
        }

        @Override
        public int hashCode() {
            int result = Arrays.hashCode(state);
            result = 31 * result + action;
            return result;
        }
    }
}
