import java.util.HashMap;
import java.util.Map;

public class TagSimulation extends Simulation {

    public TagSimulation(Environment e, Map<Integer, Agent> a) {
        super(e, a);
    }

    @Override
    protected boolean isComplete() {
        return ((TagState) env.currentState()).getIteration() >= 50;
    }

    public static void main(String[] args) {
        System.out.println("The Tag World Agent Test");
        System.out.println("-----------------------------------");
        System.out.println();

        HashMap<Integer, Agent> a = new HashMap<>();
        TagEnvironment env = new TagEnvironment();
        TagSimulation sim = new TagSimulation(env, a);
        TagState initState = TagState.getInitState(10, 10, 10, 4);

        sim.start(initState);
    }
}
