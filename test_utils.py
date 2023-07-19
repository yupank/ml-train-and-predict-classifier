from models.ml_cluster_analyser_2 import find_elbow

test_x = [2,3,4,5,6,7]
def test_find_elbow_returns_none_if_passed_empty_list():
    assert find_elbow([]) == None

def test_find_elbow_returns_last_one_for_linear_values():
    test_vals_1 = [20,30,40,50,60,70]
    test_vals_2 = [60,50,40,30,20,10]
    assert find_elbow(test_vals_1, x_vals=test_x, gradient=1) == 7
    assert find_elbow(test_vals_2,x_vals=test_x) == 7

def test_find_elbow_returns_correct_elbow_position():
    test_vals_1 = [60,40,20,10,5,0]
    test_vals_2 = [60,40,30,20,10,5]
    assert find_elbow(test_vals_1, x_vals=test_x) == 5
    assert find_elbow(test_vals_2,x_vals=test_x) == 4