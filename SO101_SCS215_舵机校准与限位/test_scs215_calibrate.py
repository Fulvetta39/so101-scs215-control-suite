import unittest

from scs215_calibrate import EXPECTED_MODEL, ServoState, packet, validate_proposal


class PacketTests(unittest.TestCase):
    def test_read_packet(self):
        self.assertEqual(packet(1, 2, (3, 2)), bytes.fromhex("ff ff 01 04 02 03 02 f3"))


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.states = [
            ServoState(i, EXPECTED_MODEL, 500, 0, 1023, 0) for i in range(1, 7)
        ]
        self.proposal = {
            "model": EXPECTED_MODEL,
            "servos": [
                {
                    "id": i,
                    "old_min": 0,
                    "old_max": 1023,
                    "proposed_min": 10,
                    "proposed_max": 1010,
                }
                for i in range(1, 7)
            ],
        }

    def test_valid_proposal(self):
        validate_proposal(self.states, self.proposal)

    def test_rejects_torque_on(self):
        states = list(self.states)
        states[2] = ServoState(3, EXPECTED_MODEL, 500, 0, 1023, 1)
        with self.assertRaisesRegex(RuntimeError, "扭矩仍开启"):
            validate_proposal(states, self.proposal)

    def test_rejects_position_outside_new_limits(self):
        states = list(self.states)
        states[0] = ServoState(1, EXPECTED_MODEL, 5, 0, 1023, 0)
        with self.assertRaisesRegex(RuntimeError, "不在提议限位"):
            validate_proposal(states, self.proposal)


if __name__ == "__main__":
    unittest.main()
