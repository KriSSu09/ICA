public class TagEnvironment extends Environment {

    public Percept getPercept(Agent agent) {
        var s = (TagState) state;
        return new TagPercept(s, agent);
    }

}
