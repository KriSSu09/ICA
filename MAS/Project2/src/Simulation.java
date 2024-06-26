import java.util.HashMap;
import java.util.Map;

/**
 * The top-level class for an agent simulation. This can be used for either
 * single or multi-agent simulations.
 */
public abstract class Simulation {

    protected HashMap<Integer, Agent> agents;
    protected Environment env;
    public static final Blackboard blackboard = new Blackboard();

    /**
     * Constructs a new simulation. Initializes the agent(or agents vector) and
     * the environment.
     */
    public Simulation(Environment e, Map<Integer, Agent> a) {
        agents = (HashMap<Integer, Agent>) a;
        env = e;
    }

    /**
     * Runs the simulation starting from a given state. This consists of a
     * sense-act loop for the/(each) agent. An alternative approach would be to
     * allow the agent to decide when it will sense and act.
     */
    public void start(State initState) {
        env.setInitialState(initState);
        env.currentState().display();

        while (!isComplete()) {
            for (Agent agent : ((TagState) env.currentState()).getAgents()) {
                Percept p = env.getPercept(agent);
                agent.see(p);
                Action action = agent.selectAction();
                env.updateState(agent, action);
                blackboard.writePosition(agent);
            }
            env.currentState().display();
            ((TagState) env.currentState()).incrementIteration();
        }
        System.out.println("END of simulation");
        System.out.println("Scores (It Score, Not It Score)");
        for (Agent a : ((TagState) env.currentState()).getAgents()) {
            TagAgent agent = (TagAgent) a;
            System.out.println(agent.getId() + " : " + agent.getAgentState().getItScore() + " , "
                    + agent.getAgentState().getNonItScore());
        }
    }

    /**
     * Is the simulation over? Returns true if it is, otherwise false.
     */
    protected abstract boolean isComplete();

}
