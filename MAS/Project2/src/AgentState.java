public class AgentState {
    private Position position;
    private boolean isIt;
    private int canTag;
    private int itScore;
    private int nonItScore;

    public AgentState(Position position, boolean isIt) {
        this.position = position;
        this.isIt = isIt;
        this.canTag = 0;
        itScore = 0;
        nonItScore = 0;
    }

    public Position getPosition() {
        return position;
    }

    public void setPosition(Position position) {
        this.position = position;
    }

    public boolean isIt() {
        return isIt;
    }

    public void setIt(boolean isIt) {
        this.isIt = isIt;
    }

    public boolean canTag() {
        return canTag == 0;
    }

    public void setCanTag(int canTag) {
        this.canTag = canTag;
    }

    public void decrementCanTag() {
        canTag--;
    }

    public void incrementScore() {
        if (isIt) {
            itScore++;
        } else {
            nonItScore++;
        }
    }

    public int getItScore() {
        return itScore;
    }

    public int getNonItScore() {
        return nonItScore;
    }

    public void decrementNonItScore() {
        nonItScore--;
    }
}
