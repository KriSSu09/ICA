public record Position(int x, int y) {

    public double distanceTo(Position otherPosition) {
        int dx = x - otherPosition.x;
        int dy = y - otherPosition.y;
        return Math.abs(dx) + Math.abs(dy);
    }
}
