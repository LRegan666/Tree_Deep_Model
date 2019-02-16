import numpy as np


def candidates_generator(state, root, k, model):
    Q, A = [root], []
    while Q:
        for node in Q:
            if node.item_id is not None:
                A.append(node)
                Q.remove(node)
        probs = []
        for node in Q:
            data = state + (np.array([[node.val]]), np.array([[0]]))
            prob = model.predict(data)
            print(prob[0])
            probs.append(prob[0])
        prob_list = list(zip(Q, probs))
        prob_list = sorted(prob_list, key=lambda x: x[1], reverse=True)
        I = []
        if len(prob_list) > k:
            for i in range(k):
                I.append(prob_list[i][0])
        else:
            for p in prob_list:
                I.append(p[0])
        Q = []
        while I:
            node = I.pop()
            if node.left:
                Q.append(node.left)
            if node.right:
                Q.append(node.right)
    probs = []
    for leaf in A:
        data = state + (np.array([[leaf.item_id]]), np.array([[1]]))
        prob = model.predict(data)
        probs.append(prob[0])
    prob_list = list(zip(A, probs))
    prob_list = sorted(prob_list, key=lambda x: x[1], reverse=True)
    A = []
    for i in range(k):
        A.append(prob_list[i][0].item_id)
    return A


def metrics_count(data, root, k, model):
    precision_rate, recall_rate, fm_rate, novelty_rate, num = 0, 0, 0, 0, 0
    for items in data:
        size = items.shape[0]
        for i in range(1):
            cands = candidates_generator((items[i][None, :],), root, k, model)
            item_clip = list(set(items[i][items[i] != -2].tolist()))
            m, g = len(cands), len(item_clip)
            for item in item_clip:
                if item in cands:
                    cands.remove(item)
            n = len(cands)
            p_rate, r_rate, n_rate = float(m - n) / m, float(m - n) / g, float(n) / k
            f_rate = (2 * p_rate * r_rate) / (p_rate + r_rate)
            precision_rate += p_rate
            recall_rate += r_rate
            fm_rate += f_rate
            novelty_rate += n_rate
            num += 1
    precision_rate = float(precision_rate * 100) / num
    recall_rate = float(recall_rate * 100) / num
    fm_rate = float(fm_rate * 100) / num
    novelty_rate = float(novelty_rate * 100) / num
    print("================================= Performance Statistic =================================")
    print("Precision rate: {:.2f}% | Recall rate: {:.2f}% | "
          "F-Measure rate: {:.2f}% | Novelty rate: {:.2f}%".format(precision_rate, recall_rate, fm_rate, novelty_rate))

