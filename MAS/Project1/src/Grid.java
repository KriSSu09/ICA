import jade.core.Agent;

import java.io.Serializable;
import java.util.*;

public class Grid implements Serializable {
    private final static int WIDTH = 15; // Width of the grid
    private final static int HEIGHT = 15; // Height of the grid
    private final char[][] grid; // Representation of the grid
    private final Map<String, List<Integer>> bombTimers = new HashMap<>();
    private final List<Agent> agents = new ArrayList<>();

    // Constants for different entities in the grid
    public final static char INDESTRUCTIBLE_WALL = 'W';
    public final static char DESTRUCTIBLE_BLOCK = 'D';
    public final static char EMPTY = ' ';
    public final static char AGENT = 'A';
    public final static char BOMB = 'B';

    public Grid() {
        grid = new char[HEIGHT][WIDTH];
        initializeGrid();
    }

    private void initializeGrid() {
        Random rand = new Random();
        // Fill the grid with indestructible walls in a checkerboard pattern and destructible blocks
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
                if (i == 0 || i == HEIGHT - 1 || j == 0 || j == WIDTH - 1) {
                    grid[i][j] = INDESTRUCTIBLE_WALL; // Set edges to be indestructible walls
                } else if ((i % 2 == 0) && (j % 2 == 0)) {
                    grid[i][j] = INDESTRUCTIBLE_WALL; // Alternating pattern within the grid
                } else if ((i % 2 != 0) && (j % 2 != 0) && rand.nextBoolean()) {
                    grid[i][j] = DESTRUCTIBLE_BLOCK; // Randomly place destructible blocks
                } else {
                    grid[i][j] = EMPTY; // Set empty spaces
                }
            }
        }
    }

    // Method for agents to register themselves
    public synchronized void registerAgent(BomberAgent agent) {
        agents.add(agent);
        if (agents.size() == 4) {
            placeAgents();
        }
    }

    public synchronized void placeAgents() {
        for (Agent agent : this.agents) {
            if (agent instanceof BomberAgent bomberAgent) {
                int x = bomberAgent.getX();
                int y = bomberAgent.getY();
                grid[x][y] = AGENT;
            }
        }
    }

    // Updates the grid location of an agent or bomb
    public void updatePosition(int oldX, int oldY, int newX, int newY, char type) {
        if (grid[newX][newY] != INDESTRUCTIBLE_WALL) {
            grid[oldX][oldY] = EMPTY;
            grid[newX][newY] = type;
        }
    }

    // Check if a position is valid for an agent to move to
    public boolean isValidMove(int x, int y) {
        return x >= 0 && x < HEIGHT && y >= 0 && y < WIDTH && grid[x][y] == EMPTY;
    }

    // Places a bomb at the specified position with a countdown
    public void placeBomb(int x, int y, int countdown, int newAgentX, int newAgentY, Agent placingAgent) {
        updatePosition(x, y, x, y, BOMB);
        int placingAgentIndex = agents.indexOf(placingAgent);
        bombTimers.put(x + "," + y, Arrays.asList(countdown, placingAgentIndex));
        updatePosition(newAgentX, newAgentY, newAgentX, newAgentY, AGENT);
    }

    // Updates the bomb timers and triggers explosions
    public void updateBombs() {
        List<String> explodedBombs = new ArrayList<>();
        for (String key : bombTimers.keySet()) {
            List<Integer> bombInfo = bombTimers.get(key);
            int timeLeft = bombInfo.get(0);
            timeLeft--;
            if (timeLeft <= 0) {
                explodedBombs.add(key);
            } else {
                bombInfo.set(0, timeLeft);
                bombTimers.put(key, bombInfo);
            }
        }

        for (String bomb : explodedBombs) {
            int x = Integer.parseInt(bomb.split(",")[0]);
            int y = Integer.parseInt(bomb.split(",")[1]);
            List<Integer> bombInfo = bombTimers.get(bomb);
            bombTimers.remove(bomb);
            explodeBomb(x, y, bombInfo.get(1));
        }
    }

    // Handles the explosion of a bomb at position (x, y)
    private synchronized void explodeBomb(int x, int y, int placingAgentIndex) {
        // Reset the bomb position to empty
        grid[x][y] = EMPTY;

        // Explode in a cross pattern
        int[] dx = {-1, 1, 0, 0};
        int[] dy = {0, 0, -1, 1};

        for (int direction = 0; direction < dx.length; direction++) {
            int nx = x, ny = y;
            while (true) {
                nx += dx[direction];
                ny += dy[direction];
                // Stop explosion if it hits an indestructible wall or the edge of the grid
                if (nx < 0 || nx >= HEIGHT || ny < 0 || ny >= WIDTH || grid[nx][ny] == INDESTRUCTIBLE_WALL) {
                    break;
                }
                // Trigger another bomb immediately if hit
                if (grid[nx][ny] == BOMB) {
                    explodeBomb(nx, ny, placingAgentIndex); // Recursive call to handle chain reaction
                    break;
                }
                // Clear the cell if it is a destructible block or another bomb
                if (grid[nx][ny] == DESTRUCTIBLE_BLOCK) {
                    BomberAgent placingAgent = null;
                    for (int i = 0; i < agents.size(); i++) {
                        if (i == placingAgentIndex) {
                            placingAgent = (BomberAgent) agents.get(i);
                            break;
                        }
                    }
                    placingAgent.updateScore(20);
                    updatePosition(nx, ny, nx, ny, EMPTY);
                    break;
                }
                // If there is an agent, handle agent death
                if (grid[nx][ny] == AGENT) {
                    removeAgent(nx, ny, placingAgentIndex);
                    break;
                }
            }
        }
        for (Agent agent : agents) {
            if (agent instanceof BomberAgent bomberAgent && bomberAgent.isAlive()) {
                bomberAgent.updateScore(10);
            }
        }
    }

    // Removes an agent from the grid
    public synchronized void removeAgent(int x, int y, int placingAgentIndex) {
        BomberAgent placingAgent = null;
        for (int i = 0; i < agents.size(); i++) {
            if (i == placingAgentIndex) {
                placingAgent = (BomberAgent) agents.get(i);
                break;
            }
        }
        for (Agent agent : agents) {
            if (agent instanceof BomberAgent bomberAgent) {
                if (bomberAgent.getX() == x && bomberAgent.getY() == y) {
                    bomberAgent.updateScore(-50);
                    bomberAgent.die();
                    if (agents.indexOf(agent) != placingAgentIndex && placingAgent != null && placingAgent.isAlive()) {
                        placingAgent.updateScore(100);
                    }
                    break;
                }
            }
        }
        updatePosition(x, y, x, y, EMPTY);
    }

    // Method to check if the game is over (one or no agents left)
    public synchronized boolean isGameOver() {
        int agentCount = 0;
        for (Agent agent : agents) {
            if (agent instanceof BomberAgent bomberAgent && bomberAgent.isAlive()) {
                agentCount++;
            }
        }
        return agentCount <= 1;
    }

    // Method to print the grid - for debugging purposes
    public void printGrid() {
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
                System.out.print(grid[i][j] + " ");
            }
            System.out.println();
        }
        System.out.println();
    }

    // Method to get the character at a specific cell
    public char getCell(int x, int y) {
        return grid[x][y];
    }

    // Methods to get grid dimensions
    public int getWidth() {
        return WIDTH;
    }

    public int getHeight() {
        return HEIGHT;
    }

    public List<Agent> getAgents() {
        return agents;
    }
}
