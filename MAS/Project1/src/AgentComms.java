import jade.core.AID;
import jade.core.Agent;
import jade.domain.AMSService;
import jade.domain.FIPAAgentManagement.AMSAgentDescription;
import jade.domain.FIPAAgentManagement.SearchConstraints;
import jade.domain.FIPAException;
import jade.lang.acl.ACLMessage;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class AgentComms {
    // Method to get all BomberAgent instances
    public static List<AMSAgentDescription> get_all_bomber_agents(Agent agent) {
        SearchConstraints sc = new SearchConstraints();
        sc.setMaxResults((long) -1);

        try {
            return Arrays.stream(AMSService.search(agent, new AMSAgentDescription(), sc))
                    .filter(a -> a.getName().getAllUserDefinedSlot().getProperty("JADE-agent-classname", "").equals("agents.BomberAgent"))
                    .collect(Collectors.toList());
        } catch (FIPAException e) {
            e.printStackTrace();
            return null;
        }
    }

    // Method to broadcast a message to all BomberAgent instances
    public static void broadcastToBomberAgents(Agent sender, String content) {
        List<AMSAgentDescription> bomberAgents = get_all_bomber_agents(sender);
        if (bomberAgents != null) {
            bomberAgents.forEach(agentDescription -> {
                String agentLocalName = agentDescription.getName().getLocalName();
                if (!agentLocalName.equals(sender.getLocalName())) {
                    ACLMessage msg = new ACLMessage(ACLMessage.INFORM);
                    msg.addReceiver(new AID(agentLocalName, AID.ISLOCALNAME));
                    msg.setContent(content);
                    sender.send(msg);
                }
            });
        }
    }
}
