import jade.core.Agent;
import jade.core.behaviours.TickerBehaviour;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class GameControllerAgent extends Agent {
    private Grid grid;
    private int ticks = 0;

    @Override
    protected void setup() {
        Object[] args = getArguments();
        if (args != null && args.length == 1) {
            grid = (Grid) args[0];
        } else {
            throw new IllegalArgumentException("Arguments required: grid");
        }

        // Game control logic
        addBehaviour(new TickerBehaviour(this, 10) { // run every second
            @Override
            protected void onTick() {
                ticks++;
                grid.printGrid();
                grid.updateBombs();
                if (ticks % 5 == 0) {
                    AtomicInteger leastMoves = new AtomicInteger(Integer.MAX_VALUE);
                    AtomicReference<List<BomberAgent>> leastMovingAgents = new AtomicReference<>(new ArrayList<>());
                    grid.getAgents().forEach(agent -> {
                        if (agent instanceof BomberAgent bomberAgent) {
                            if (bomberAgent.getMoveCount() < leastMoves.get()) {
                                leastMoves.set(bomberAgent.getScore());
                                leastMovingAgents.get().clear();
                                leastMovingAgents.get().add(bomberAgent);
                            } else if (bomberAgent.getMoveCount() == leastMoves.get()) {
                                leastMovingAgents.get().add(bomberAgent);
                            }
                        }
                    });
                    leastMovingAgents.get().forEach(agent -> agent.updateScore(-5));
                }
                if (grid.isGameOver()) {
                    System.out.println("Game Over!");
                    System.out.println(ticks + " ticks elapsed.");
                    grid.printGrid();
                    AtomicInteger bestScore = new AtomicInteger(Integer.MIN_VALUE);
                    AtomicReference<BomberAgent> bestAgent = new AtomicReference<>();
                    grid.getAgents().forEach(agent -> {
                        if (agent instanceof BomberAgent bomberAgent) {
                            if (bomberAgent.getScore() > bestScore.get()) {
                                bestScore.set(bomberAgent.getScore());
                                bestAgent.set(bomberAgent);
                            }
                            bomberAgent.die(); // Stop the agent
                        }
                    });
                    System.out.println("Best agent: " + bestAgent.get().getLocalName() + " with score " + bestScore.get());
                    bestAgent.get().saveQTable("qtable.ser");
                    doDelete(); // Stop this agent
                }
            }
        });
    }
}