import jade.core.Profile;
import jade.core.ProfileImpl;
import jade.core.Runtime;
import jade.wrapper.AgentContainer;
import jade.wrapper.AgentController;
import jade.wrapper.StaleProxyException;


public class Main {
    public static void main(String[] args) {
        // Setup JADE runtime
        Runtime rt = Runtime.instance();
        Profile profile = new ProfileImpl();
        profile.setParameter(Profile.MAIN_HOST, "localhost");
        profile.setParameter(Profile.MAIN_PORT, "1099");
        AgentContainer mainContainer = rt.createMainContainer(profile);

        try {
            // Prepare the grid
            Grid grid = new Grid();

            // Specific corner positions
            int[][] corners = {
                    {1, 1}, // Top-left corner
                    {1, grid.getWidth() - 2}, // Top-right corner
                    {grid.getHeight() - 2, 1}, // Bottom-left corner
                    {grid.getHeight() - 2, grid.getWidth() - 2} // Bottom-right corner
            };

            // Add a game controller agent for managing game logic
            AgentController gameController = mainContainer.createNewAgent("GameController", "GameControllerAgent", new Object[]{grid});

            // Create agents in the corners
            String[] agentNames = {"Bomber1", "Bomber2", "Bomber3", "Bomber4"};
            for (int i = 0; i < corners.length; i++) {
                int startX = corners[i][0];
                int startY = corners[i][1];
                Object[] agentArgs = {startX, startY, grid, true};
                AgentController ac = mainContainer.createNewAgent(agentNames[i], "BomberAgent", agentArgs);
                ac.start();
            }

            gameController.start();

        } catch (StaleProxyException e) {
            e.printStackTrace();
        }
    }
}