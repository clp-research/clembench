
def get_size(grid):
    rows = grid.strip().split('\n')

    row_size = len(rows)
    column_size = 0
    for r_index in range(0, len(rows)):
        target_cells = rows[r_index].split(' ')
        column_size = len(target_cells)
        break

    return row_size, column_size

def evaluate(target, generated):

    if get_size(target) != get_size(generated):
        return 0.0, 0.0, 0.0

    target_rows = target.strip().split('\n')
    generated_rows = generated.strip().split('\n')

    recall_counter = 0
    total_recall_counter = 0

    precision_counter = 0
    total_precision_counter = 0

    for r_index in range(0, len(target_rows)):

        target_cells = target_rows[r_index].split(' ')
        generated_cells = generated_rows[r_index].split(' ')

        for c_index in range(0, len(target_cells)):

            if target_cells[c_index] != '▢':
                total_recall_counter += 1

                if target_cells[c_index].lower() == generated_cells[c_index].lower():
                    recall_counter += 1

            if generated_cells[c_index] != '▢':
                total_precision_counter += 1

                if target_cells[c_index].lower() == generated_cells[c_index].lower():
                    precision_counter += 1

    recall = round(recall_counter/float(total_recall_counter), 4)
    precision = round(precision_counter / float(total_precision_counter), 4)

    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = (2*precision*recall) / (precision+recall)
    f1 = round(f1, 4)

    precision = round(100*precision, 0)
    recall = round(100 * recall, 0)
    f1 = round(100 * f1, 0)

    return precision, recall, f1

def calculate_flipped_pixels(previous, current):
    target_rows = previous.strip().split('\n')
    generated_rows = current.strip().split('\n')

    flipped_counter = 0

    for r_index in range(0, len(target_rows)):

        target_cells = target_rows[r_index].split(' ')
        generated_cells = generated_rows[r_index].split(' ')

        for c_index in range(0, len(target_cells)):

            if target_cells[c_index].lower() != generated_cells[c_index].lower():
                flipped_counter += 1

    return flipped_counter
