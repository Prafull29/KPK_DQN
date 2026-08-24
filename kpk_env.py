import numpy as np
import gymnasium as gym
from gymnasium import spaces
import chess
import chess.syzygy


class KPKEnv(gym.Env):
    
    metadata = {
        "render_modes": ["human"]
    }

    # CONSTANTS

    WHITE = 0
    BLACK = 1

    # Actions

    KING_UP = 0
    KING_DOWN = 1
    KING_LEFT = 2
    KING_RIGHT = 3
    KING_UP_LEFT = 4
    KING_UP_RIGHT = 5
    KING_DOWN_LEFT = 6
    KING_DOWN_RIGHT = 7

    PAWN_ONE = 8
    PAWN_TWO = 9

    # King movement directions

    KING_DIRECTIONS = [
        (0, 1),     # 0: up
        (0, -1),    # 1: down
        (-1, 0),    # 2: left
        (1, 0),     # 3: right
        (-1, 1),    # 4: up-left
        (1, 1),     # 5: up-right
        (-1, -1),   # 6: down-left
        (1, -1)     # 7: down-right
    ]

    # INITIALIZATION

    def __init__(self,render_mode=None,tablebase_path="/kaggle/input/datasets/prafull29/kpk-dqn-project"):

        super().__init__()

        self.render_mode = render_mode

        # Syzygy tablebase

        self.tablebase = chess.syzygy.open_tablebase(
           "/kaggle/input/datasets/prafull29/kpk-dqn-project"
        )

        # STATE SPACE

        self.observation_space = spaces.MultiDiscrete(
            [64, 64, 64, 2]
        )

        # ACTION SPACE

        self.action_space = spaces.Discrete(10)

        # Current state

        self.wk = None
        self.wp = None
        self.bk = None
        self.turn = None

        # Episode information

        self.done = False
        self.winner = None

    # RESET

    def white_can_win(self, wk, wp, bk, turn):

        board = chess.Board(None)

        board.set_piece_at(
            wk,
            chess.Piece(chess.KING, chess.WHITE)
        )

        board.set_piece_at(
            wp,
            chess.Piece(chess.PAWN, chess.WHITE)
        )

        board.set_piece_at(
            bk,
            chess.Piece(chess.KING, chess.BLACK)
        )

        board.turn = (
            chess.WHITE
            if turn == self.WHITE
            else chess.BLACK
        )

        try:

            wdl = self.tablebase.probe_wdl(board)

        except chess.syzygy.MissingTableError:

            return False

        # WDL is from the side-to-move perspective.
        #
        # White to move:
        # positive WDL = White winning
        #
        # Black to move:
        # negative WDL = White winning

        if turn == self.WHITE:
            return wdl > 0

        else:
            return wdl < 0

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        while True:

            # Random legal KPK position

            self.wk = int(
                self.np_random.integers(0, 64)
            )

            self.wp = int(
                self.np_random.integers(8, 56)
            )

            self.bk = int(
                self.np_random.integers(0, 64)
            )

            self.turn = int(
                self.np_random.integers(0, 2)
            )

            # Must be legal

            if not self.is_legal_position(
                self.wk,
                self.wp,
                self.bk,
                self.turn
            ):
                continue

            # IMPORTANT:
            # Only keep positions where
            # White has a theoretical win.

            if not self.white_can_win(
                self.wk,
                self.wp,
                self.bk,
                self.turn
            ):
                continue

            break

        self.done = False
        self.winner = None

        # If Black starts, let optimal Black
        # make the first move.

        if self.turn == self.BLACK:

            black_move = self.get_black_optimal_move()

            if black_move is None:

                self.done = True
                self.winner = "DRAW"

            else:

                self.apply_black_move(
                    black_move
                )

                if self.black_captured_pawn():

                    self.done = True
                    self.winner = "DRAW"

                else:

                    self.turn = self.WHITE

                    terminated, reward, result = (
                        self.check_terminal()
                    )

                    if terminated:

                        self.done = True
                        self.winner = result

        return (
            self.get_state(),
            self.get_info()
        )

    # STATE

    def get_state(self):

        return np.array(
            [
                self.wk,
                self.wp,
                self.bk,
                self.turn
            ],
            dtype=np.int32
        )

    # SQUARE CONVERSION

    def xy_to_square(self, x, y):

        if x < 0 or x > 7:
            return None

        if y < 0 or y > 7:
            return None

        return y * 8 + x

    def square_to_xy(self, square):

        x = square % 8
        y = square // 8

        return x, y

    # SQUARE NAME

    def square_name(self, square):

        if square is None:
            return "None"

        x, y = self.square_to_xy(square)

        file_name = chr(
            ord("a") + x
        )

        rank_name = str(y + 1)

        return file_name + rank_name

    # CONVERT SQUARE TO NUMBER

    def square(self, name):

        if len(name) != 2:
            raise ValueError(
                "Invalid square name."
            )

        file_char = name[0]
        rank_char = name[1]

        x = ord(file_char) - ord("a")

        try:
            y = int(rank_char) - 1
        except ValueError:
            raise ValueError(
                "Invalid rank."
            )

        if x < 0 or x > 7:
            raise ValueError(
                "Invalid file."
            )

        if y < 0 or y > 7:
            raise ValueError(
                "Invalid rank."
            )

        return y * 8 + x

    # KING ATTACKS

    def king_attacks(self, king_square, target_square):

        kx, ky = self.square_to_xy(
            king_square
        )

        tx, ty = self.square_to_xy(
            target_square
        )

        dx = abs(kx - tx)
        dy = abs(ky - ty)

        return max(dx, dy) == 1

    # WHITE PAWN ATTACKS

    def pawn_attacks(self,pawn_square,target_square):

        px, py = self.square_to_xy(
            pawn_square
        )

        tx, ty = self.square_to_xy(
            target_square
        )

        return (
            ty == py + 1
            and abs(tx - px) == 1
        )

    # WHITE IN CHECK

    def white_in_check(self):

        return self.king_attacks(
            self.bk,
            self.wk
        )

    # BLACK IN CHECK

    def black_in_check(self):

        # White King attacks Black King

        if self.king_attacks(
            self.wk,
            self.bk
        ):
            return True

        # White Pawn attacks Black King

        if self.wp is not None:

            if self.pawn_attacks(
                self.wp,
                self.bk
            ):
                return True

        return False

    # CURRENT PLAYER IN CHECK

    def is_in_check(self):

        if self.turn == self.WHITE:

            return self.white_in_check()

        return self.black_in_check()

    # KINGS ADJACENT

    def kings_adjacent(self,wk,bk):

        return self.king_attacks(
            wk,
            bk
        )

    # LEGAL POSITION

    def is_legal_position(self,wk,wp,bk,turn):

        # Range checks

        if not (0 <= wk <= 63):
            return False

        if not (8 <= wp <= 55):
            return False

        if not (0 <= bk <= 63):
            return False

        if turn not in [
            self.WHITE,
            self.BLACK
        ]:
            return False

        # Pieces cannot occupy same square

        if wk == wp:
            return False

        if wk == bk:
            return False

        if wp == bk:
            return False

        # Kings cannot be adjacent

        if self.kings_adjacent(wk, bk):
            return False

        # Check whether the position has an impossible check.

        old_wk = self.wk
        old_wp = self.wp
        old_bk = self.bk
        old_turn = self.turn

        self.wk = wk
        self.wp = wp
        self.bk = bk
        self.turn = turn

        if turn == self.WHITE:

            if self.black_in_check():

                self.wk = old_wk
                self.wp = old_wp
                self.bk = old_bk
                self.turn = old_turn

                return False

        else:

            if self.white_in_check():

                self.wk = old_wk
                self.wp = old_wp
                self.bk = old_bk
                self.turn = old_turn

                return False

        # Restore environment state

        self.wk = old_wk
        self.wp = old_wp
        self.bk = old_bk
        self.turn = old_turn

        return True

    # MOVE WHITE KING

    def move_king(self,wk,action):

        if action < 0 or action > 7:
            return None

        x, y = self.square_to_xy(wk)

        dx, dy = self.KING_DIRECTIONS[action]

        new_x = x + dx
        new_y = y + dy

        return self.xy_to_square(
            new_x,
            new_y
        )

    # MOVE WHITE PAWN
    
    def move_pawn(self,wp,action):

        if wp is None:
            return None

        x, y = self.square_to_xy(wp)

        # One square

        if action == self.PAWN_ONE:

            new_y = y + 1

            return self.xy_to_square(
                x,
                new_y
            )

        # Two squares

        if action == self.PAWN_TWO:

            if y != 1:
                return None

            new_y = y + 2

            return self.xy_to_square(
                x,
                new_y
            )

        return None

    # LEGAL WHITE KING MOVE

    def is_legal_white_king_move(self,action):

        new_wk = self.move_king(
            self.wk,
            action
        )

        if new_wk is None:
            return False

        # Cannot move onto own pawn

        if new_wk == self.wp:
            return False

        # Cannot move onto Black King
      
        if new_wk == self.bk:
            return False

        # Black King attacks destination

        if self.king_attacks(
            self.bk,
            new_wk
        ):
            return False

        return True

    # LEGAL WHITE PAWN MOVE

    def is_legal_white_pawn_move(self,action):

        if self.wp is None:
            return False

        new_wp = self.move_pawn(
            self.wp,
            action
        )
    
        if new_wp is None:
            return False

        # Pawn cannot move onto either king

        if new_wp == self.wk:
            return False

        if new_wp == self.bk:
            return False

        # Two-square pawn move

        if action == self.PAWN_TWO:

            one_step = self.wp + 8

            # First square must be empty
            if one_step == self.wk:
                return False

            if one_step == self.bk:
                return False

        return True

    # ALL LEGAL WHITE ACTIONS

    def get_valid_actions(self):

        valid_actions = []

        # White King
    
        for action in range(8):

            if self.is_legal_white_king_move(
                action
            ):
                valid_actions.append(action)

        # White Pawn: one square

        if self.is_legal_white_pawn_move(
            self.PAWN_ONE
        ):
            valid_actions.append(
                self.PAWN_ONE
            )

        # White Pawn: two squares

        if self.is_legal_white_pawn_move(
            self.PAWN_TWO
        ):
            valid_actions.append(
                self.PAWN_TWO
            )

        return valid_actions

    # ACTION MASK

    def get_action_mask(self):

        mask = np.zeros(
            10,
            dtype=np.float32
        )

        for action in self.get_valid_actions():

            mask[action] = 1.0

        return mask

    # APPLY WHITE KING MOVE

    def apply_white_king_move(self,action):

        self.wk = self.move_king(
            self.wk,
            action
        )

    # APPLY WHITE PAWN MOVE

    def apply_white_pawn_move(self,action):

        self.wp = self.move_pawn(
            self.wp,
            action
        )

    # APPLY WHITE ACTION

    def apply_white_action(self,action):

        if action <= 7:

            self.apply_white_king_move(
                action
            )

        else:

            self.apply_white_pawn_move(
                action
            )

    # PAWN PROMOTION

    def pawn_promoted(self):

        if self.wp is None:
            return False

        _, pawn_y = self.square_to_xy(
            self.wp
        )

        return pawn_y == 7

    # BLACK KING LEGAL MOVE

    def is_legal_black_king_move(self,from_square,to_square):

        # Must be one-square king movement

        if not self.king_attacks(
            from_square,
            to_square
        ):
            return False

        # Cannot move onto White King

        if to_square == self.wk:
            return False

        # White King attacks destination

        if self.king_attacks(
            self.wk,
            to_square
        ):
            return False

        if self.wp is not None:

            if to_square != self.wp:

                if self.pawn_attacks(
                    self.wp,
                    to_square
                ):
                    return False

        return True

    # BLACK LEGAL MOVES

    def get_black_legal_moves(self):

        legal_moves = []

        bx, by = self.square_to_xy(
            self.bk
        )

        for dx, dy in self.KING_DIRECTIONS:

            new_x = bx + dx
            new_y = by + dy

            new_bk = self.xy_to_square(
                new_x,
                new_y
            )

            if new_bk is None:
                continue

            if self.is_legal_black_king_move(
                self.bk,
                new_bk
            ):
                legal_moves.append(
                    new_bk
                )

        return legal_moves

    # BLACK OPTIMAL MOVE

    def get_black_optimal_move(self):

        legal_moves = self.get_black_legal_moves()

        if len(legal_moves) == 0:
            return None

        best_move = None
        best_wdl = -10
        best_dtz = None

        # Current KPK board
        board = chess.Board(None)

        board.set_piece_at(
            self.wk,
            chess.Piece(chess.KING, chess.WHITE)
        )

        board.set_piece_at(
            self.wp,
            chess.Piece(chess.PAWN, chess.WHITE)
        )

        board.set_piece_at(
            self.bk,
            chess.Piece(chess.KING, chess.BLACK)
        )

        board.turn = chess.BLACK

        for move in legal_moves:

            chess_move = chess.Move(
                self.bk,
                move
            )

            board.push(chess_move)

            try:

                wdl = -self.tablebase.probe_wdl(board)
                dtz = -self.tablebase.probe_dtz(board)

            except chess.syzygy.MissingTableError:

                board.pop()
                continue

            board.pop()

            if wdl > best_wdl:

                best_wdl = wdl
                best_dtz = dtz
                best_move = move

            elif wdl == best_wdl:

                if (
                    best_dtz is None
                    or abs(dtz) < abs(best_dtz)
                ):

                    best_dtz = dtz
                    best_move = move

        return best_move

    # APPLY BLACK MOVE

    def apply_black_move(self,new_bk):
        self.bk = new_bk

    # BLACK CAPTURED PAWN

    def black_captured_pawn(self):
        return (
            self.wp is not None
            and self.bk == self.wp
        )

    # BLACK STALEMATE

    def is_black_stalemate(self):
        return (
            self.turn == self.BLACK
            and len(self.get_black_legal_moves()) == 0
            and not self.black_in_check()
        )


    def is_black_checkmate(self):
        return (
            self.turn == self.BLACK
            and len(self.get_black_legal_moves()) == 0
            and self.black_in_check()
        )


    def is_white_stalemate(self):
        return (
            self.turn == self.WHITE
            and len(self.get_valid_actions()) == 0
            and not self.white_in_check()
        )


    def is_white_checkmate(self):
        return (
            self.turn == self.WHITE
            and len(self.get_valid_actions()) == 0
            and self.white_in_check()
        )

    # TERMINAL STATE

    def check_terminal(self):

        # Black captured the pawn
        # K vs K is a draw

        if self.black_captured_pawn():

            return (
                True,
                0.0,
                "DRAW"
            )

        # White promoted

        if self.pawn_promoted():

            return (
                True,
                1.0,
                "WHITE_WIN"
            )

        # White to move

        if self.turn == self.WHITE:

            if self.is_white_stalemate():

                return (
                    True,
                    0.0,
                    "DRAW"
                )

        # Black to move

        else:

            if self.turn == self.BLACK:

                if self.is_black_checkmate():
                    return True, 1.0, "WHITE_WIN"

                if self.is_black_stalemate():
                    return True, 0.0, "DRAW"

        return (
            False,
            0.0,
            None
        )

    # STEP

    def step(self, action):

        # Game already over
    
        if self.done:

            return (
                self.get_state(),
                0.0,
                True,
                False,
                self.get_info()
            )

        # Agent must act as White

        if self.turn != self.WHITE:
            self.turn = self.WHITE

        # Check action

        valid_actions = self.get_valid_actions()

        if action not in valid_actions:

            return (
                self.get_state(),
                -0.1,
                False,
                False,
                self.get_info()
            )

        # WHITE MOVE

        self.apply_white_action(action)

        # Change turn to Black

        self.turn = self.BLACK

        # Check if White's move ended the game

        terminated, reward, result = (
            self.check_terminal()
        )

        if terminated:

            self.done = True
            self.winner = result

            return (
                self.get_state(),
                reward,
                True,
                False,
                self.get_info()
            )

        # BLACK OPTIMAL MOVE

        black_move = self.get_black_optimal_move()

        # Black has no legal move

        if black_move is None:

            terminated, reward, result = (
                self.check_terminal()
            )

            self.done = terminated
            self.winner = result

            return (
                self.get_state(),
                reward,
                terminated,
                False,
                self.get_info()
            )

        # BLACK MOVE

        self.apply_black_move(
            black_move
        )

        # Check if Black captured pawn

        if self.black_captured_pawn():

            self.done = True
            self.winner = "DRAW"

            return (
                self.get_state(),
                0.0,
                True,
                False,
                self.get_info()
            )

        # Back to White

        self.turn = self.WHITE

        # Check terminal

        terminated, reward, result = (
            self.check_terminal()
        )

        if terminated:

            self.done = True
            self.winner = result
        
        # Black has finished its move.
        # Next decision belongs to White.

        # Return

        if not terminated:
            self.turn = self.WHITE

        return (
            self.get_state(),
            reward,
            terminated,
            False,
            self.get_info()
        )

    # INFO

    def get_info(self):

        if self.turn == self.WHITE:

            valid_actions = self.get_valid_actions()

            action_mask = self.get_action_mask()

        else:

            valid_actions = []

            action_mask = np.zeros(
                10,
                dtype=np.float32
            )

        return {

            "white_king":
                self.wk,

            "white_pawn":
                self.wp,

            "black_king":
                self.bk,

            "turn":
                self.turn,

            "turn_name":
                "White"
                if self.turn == self.WHITE
                else "Black",

            "valid_actions":
                valid_actions,

            "action_mask":
                action_mask,

            "white_in_check":
                self.white_in_check(),

            "black_in_check":
                self.black_in_check(),

            "winner":
                self.winner,

            "done":
                self.done
        }

    # RENDER

    def render(self):

        board = [
            ["."] * 8
            for _ in range(8)
        ]

        # White King

        wx, wy = self.square_to_xy(
            self.wk
        )

        board[wy][wx] = "K"

        # White Pawn

        if self.wp is not None:

            px, py = self.square_to_xy(
                self.wp
            )

            # If Black captured the pawn, don't print it
            if self.bk != self.wp:

                board[py][px] = "P"

        # Black King

        bx, by = self.square_to_xy(
            self.bk
        )

        board[by][bx] = "k"

        # Print board

        print()
        print("  +-----------------+")

        for y in range(7, -1, -1):

            row = " ".join(
                board[y]
            )

            print(
                f"{y + 1} | {row} |"
            )

        print("  +-----------------+")
        print("    a b c d e f g h")

        print()

        print(
            "White King:",
            self.square_name(self.wk)
        )

        print(
            "White Pawn:",
            self.square_name(self.wp)
        )

        print(
            "Black King:",
            self.square_name(self.bk)
        )

        print(
            "Turn:",
            "WHITE"
            if self.turn == self.WHITE
            else "BLACK"
        )

        if self.turn == self.WHITE:

            print(
                "Valid actions:",
                self.get_valid_actions()
            )

            print(
                "Action mask:",
                self.get_action_mask()
            )

        print(
            "White in check:",
            self.white_in_check()
        )

        print(
            "Black in check:",
            self.black_in_check()
        )

        print(
            "Winner:",
            self.winner
        )

        print()

    # CLOSE

    def close(self):

        pass
