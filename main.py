from simulator import GameSimulator

if __name__ == '__main__':
    """
    Main entry point for the AI Tank Battle Simulator.

    This script initializes and runs the GameSimulator, starting the
    self-play training loop between the two AI agents.
    """
    simulator = GameSimulator()
    simulator.run()
