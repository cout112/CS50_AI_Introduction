from hashlib import new
import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count, mineCells=set()):
        self.cells = set(cells)
        self.count = count
        self.safeCells = set()
        self.mineCells = set(mineCells)


    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        return self.mineCells
        

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        return self.safeCells


    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """

        if cell in self.cells:
            self.mineCells.add(cell)
            self.count -= 1
            self.cells.remove(cell)
        
        pass


    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """

        if cell in self.cells:
            self.safeCells.add(cell)
            self.cells.remove(cell)
        pass


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

        #Keep count of iterations
        self.c = 0


    def delete_sentence(self, sentence):
        index = self.knowledge.index(sentence)
        self.knowledge.pop(index)
        pass

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)
            if len(sentence.cells) == 0:
                self.delete_sentence(sentence)
            elif len(sentence.cells) == 1:
                if sentence.count == 0:
                    cell = list(sentence.cells)[0]
                    self.mark_safe(cell)
                elif sentence.count == 1:
                    cell = list(sentence.cells)[0]
                    self.mark_mine(cell)
            

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)
            if len(sentence.cells) == 0:
                self.delete_sentence(sentence)
            elif len(sentence.cells) == 1:
                if sentence.count == 0:
                    cell = list(sentence.cells)[0]
                    self.mark_safe(cell)
                elif sentence.count == 1:
                    cell = list(sentence.cells)[0]
                    self.mark_mine(cell)
            

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        print(f"Move made: {cell}")
        self.moves_made.add(cell)
        self.safes.add(cell)

        """Mark cell as safe"""
        self.mark_safe(cell)

        #Find nearby cells
        sentenceCells = []
        mineCells = []
        for i in range(cell[0]-1, cell[0]+2):
            for j in range(cell[1]-1, cell[1]+2):
                if (i,j) == cell:
                    continue
                if (i, j) in self.safes:
                    continue
                if (i, j) in self.mines:
                    mineCells.append((i, j))
                
                if 0 <= i < self.width and 0 <= j < self.width:
                    sentenceCells.append((i,j))

        """ If count is 0, all the cells in the subset are safe """
        if count == 0:
            for cell in sentenceCells:
                self.mark_safe(cell)
        
        else:
            newSentence = Sentence(sentenceCells, count, mineCells)
            self.knowledge.append(newSentence)


        compareAgain = True
        while compareAgain:
            compareAgain = self.compare_sentences()

            safeMovements = set()
            for cell in self.safes:
                if cell not in self.moves_made:
                    safeMovements.add(cell)
            # print(f"Count: {self.c}")
            print()
            print("SENTENCES:")
            for sentence in self.knowledge:
                print(sentence)
            print()
            print("SAFE: ", safeMovements)
            print('MINES: ', self.mines)
            print()
        
            
        pass



    def check_mines_safes(self):
        """Check for known mines in knowledge"""
        somethingChanged = False
        for i, sentence in enumerate(self.knowledge):
            if sentence.count == len(sentence.cells):
                cells = list(sentence.cells)
                for cell in cells:
                    self.mark_mine(cell)
                    somethingChanged = True
            if sentence.count == 0:
                cells = list(sentence.cells)
                for cell in cells:
                    self.mark_safe(cell)
                    somethingChanged = True
        for sentence in self.knowledge:
            mineCells = []
            for cell in sentence.cells:
                if cell in self.mines:
                    mineCells.append(cell)
            for cell in mineCells:
                self.mark_mine(cell)
                somethingChanged = True
        return somethingChanged


    def compare_sentences(self):
        """Compare all sentences with each other"""
        
        self.c += 1
        

        somethingChanged = self.check_mines_safes()

        """Remove duplicates"""
        sentencesToDelete = []
        for i, sentence_1 in enumerate(self.knowledge):
            for ii, sentence_2 in enumerate(self.knowledge):
                if ii <= i:
                    continue
                if sentence_1.cells == sentence_2.cells:
                    sentencesToDelete.append(sentence_1)
        for sentence in sentencesToDelete:
            self.delete_sentence(sentence)
                    

        for i, sentence_1 in enumerate(self.knowledge):
            for ii, sentence_2 in enumerate(self.knowledge):
                if ii <= i:
                    continue

                """Find if sentence 1 is a subset of sentence 2"""
                is_1_in_2 = True
                for cell_1 in sentence_1.cells:
                    if cell_1 not in sentence_2.cells:
                        is_1_in_2 = False
                
                if is_1_in_2:
                    """Find distinct subset"""
                    distinctSet = set()
                    for cell_2 in sentence_2.cells:
                        if cell_2 not in sentence_1.cells:
                            distinctSet.add(cell_2)
                    newCount = sentence_2.count - sentence_1.count
                    if newCount == 0:
                        for cell in distinctSet:
                            self.mark_safe(cell)
                        somethingChanged = True

                    elif len(distinctSet) == 1:
                        for cell in distinctSet:
                            self.mark_mine(cell)
                        somethingChanged = True
                        
                    else:
                        exists = False
                        for cells in self.knowledge:
                            if distinctSet == cells.cells:
                                exists = True
                                break
                        if not exists:
                            sentence = Sentence(distinctSet, newCount)
                            self.knowledge.append(sentence)
                            somethingChanged = True

                """Find if sentence 2 is a subset of sentence 1"""
                is_2_in_1 = True
                for cell_2 in sentence_2.cells:
                    if cell_2 not in sentence_1.cells:
                        is_2_in_1 = False
                
                if is_2_in_1:
                    """Find distinct subset"""
                    distinctSet = set()
                    for cell_1 in sentence_1.cells:
                        if cell_1 not in sentence_2.cells:
                            distinctSet.add(cell_1)
                    newCount = sentence_1.count - sentence_2.count
                    if newCount == 0:
                        for cell in distinctSet:
                            self.mark_safe(cell)
                        somethingChanged = True

                    elif len(distinctSet) == 1:
                        for cell in distinctSet:
                            self.mark_mine(cell)
                        somethingChanged = True

                    else:
                        exists = False
                        for cells in self.knowledge:
                            if distinctSet == cells.cells:
                                exists = True
                                break
                        if not exists:
                            sentence = Sentence(distinctSet, newCount)
                            self.knowledge.append(sentence)
                            somethingChanged = True

        return somethingChanged        



    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        safeMovements = set()
        for cell in self.safes:
            if cell not in self.moves_made:
                safeMovements.add(cell)
        return next(iter(safeMovements)) if len(safeMovements) > 0 else None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        availableCells = []
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.moves_made and (i, j) not in self.mines:
                    availableCells.append((i, j))

        return availableCells[random.randrange(len(availableCells))] if len(availableCells) > 0 else None

