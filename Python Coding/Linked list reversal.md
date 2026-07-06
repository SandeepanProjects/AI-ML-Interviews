class Node:
    def __init__(self, new_data):

        self.data = new_data
        self.next = None


def reverseList(head):

    stack = []

    temp = head

    # push all nodes except the last node into stack
    while temp.next is not None:
        stack.append(temp)
        temp = temp.next

    head = temp

    # pop all the nodes and append to the linked list
    while stack:

        # append the top value of stack in list
        temp.next = stack.pop()

        temp = temp.next

    temp.next = None

    return head


def printList(node):
    while node is not None:
        print(f"{node.data}", end="")
        if node.next is not None:
            print(" -> ", end="")
        node = node.next
    print()

if __name__ == "__main__":
    # create a hard-coded linked list:
    # 1 -> 2 -> 3 -> 4 -> 5
    head = Node(1)
    head.next = Node(2)
    head.next.next = Node(3)
    head.next.next.next = Node(4)
    head.next.next.next.next = Node(5)

    head = reverseList(head)
    printList(head)