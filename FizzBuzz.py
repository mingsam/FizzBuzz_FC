def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3  # FizzBuzz
    elif i % 5 == 0:
        return 2  # Buzz
    elif i % 3 == 0:
        return 1  # Fizz
    else:
        return 0


def fizz_buzz_decoder(i, predic_str):
    return [str(i), "fizz", "buzz", "fizzbuzz"][predic_str]

# for test
# print(fizz_buzz_decoder(1,fizz_buzz_encode(1)))
# print(fizz_buzz_decoder(3,fizz_buzz_encode(3)))
# print(fizz_buzz_decoder(5,fizz_buzz_encode(5)))
# print(fizz_buzz_decoder(15,fizz_buzz_encode(15)))
