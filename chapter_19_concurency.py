import os
from threading import Thread, Event

from multiprocessing import Process, Event
# we use synchronize to access its Event class for type hinting! as the multiprocessing.Event is a function
# not a class! 
from multiprocessing import synchronize

#or async
import asyncio

import itertools
import time 



def run_heavy_op():
    time.sleep(3)
    return 45

def main_thread():
    # define the animation function
    def loading_symbol(msg: str, event: Event)-> None:
        #print loading animation until instructed to end and print the msg
        for ch in itertools.cycle('|/\\|-'):
            # use \r to display animation in place
            # use end='' to not create new lines
            # use flush=True to always update the output
            text = f'\r{ch} {msg}'
            print(text, end='',flush=True)
            
            # basically waits here until another thread says its ok to continue 
            # by settin event.set()
            if (event.wait(0.1)):
                break
        # beore we end this call lets cleanup
        blanks = ' '*len(text)
        print(f"\r{blanks}\r",end='')
        print(f'animation end')
    
    event = Event()
    thread_loading_animation = Thread(target=loading_symbol, args=("loading please wait ...", event))
    print(f'starting the loading animation process(thread):{thread_loading_animation}')
    thread_loading_animation.start()
    # running the heavy ufnction
    result=run_heavy_op()
    print('heavy_app_ended, lets end the loading animation')
    #end the loading animation
    event.set()
    # when the loading animation process is done, wait or the main thread here 
    thread_loading_animation.join()
    print (f"result gotten is {result}")
    
def main_multiprocessing():
    # define the function
    def loading_animation(msg: str, event: synchronize.Event)-> None:
        #print loading animation until instructed to end and print the msg
        for ch in itertools.cycle('|/\\|-'):
            # use \r to display animation in place
            # use end='' to not create new lines
            # use flush=True to always update the output
            text = f'\r{ch} {msg}'
            print(text, end='',flush=True)
            
            # basically waits here until another thread says its ok to continue 
            # by settin event.set()
            if (event.wait(0.1)):
                break
        # beore we end this call lets cleanup
        blanks = ' '*len(text)
        print(f"\r{blanks}\r",end='')
        print(f'animation end')
        
    event = Event()
    process_animation =Process(target=loading_animation, args=('loading please wait', event))
    # beore the heavy load process starts, lets run our loading animation!s
    print(f'process : {process_animation}')
    process_animation.start()
    result = run_heavy_op()
    # signal process that all is done here, end yourself!s
    event.set()
    process_animation.join()
    print(f'result : {result}')
    
# async is required so we can use await, and other async related machinery 
# in this function!
async def run_coroutine_animation():
    
    async def heavy_process():
        # placing anything outside of asyncio
        # will mess with the coroutine, e.g. if you use time.sleep
        # as long as its sleeping, and we havent reached await asyncio, 
        # no other coroutine gets executed, that is, no animation gets played
        # that is all coroutines get blocked! in order to block the current 
        #coroutine we use the asyncio.sleep instead of time.sleep
        # this way, all other coroutines wont get blocked!
        
        # from effective python: 
        # To understand what is happening, recall that Python code using asyncio has only
        # one flow of execution, unless you’ve explicitly started additional threads or processes.
        # That means only one coroutine executes at any point in time. Concurrency is
        # achieved by control passing from one coroutine to another.
        # time.sleep(3)
        
        # when you use `await` on a coroutine in an asynchronous context, it means that the coroutine voluntarily suspends its execution 
        # and relinquishes control to the event loop. This allows other coroutines or tasks to be executed while the awaited operation is 
        # in progress.
        # When you encounter a line of code like `await asyncio.sleep(3)` in a coroutine, the following steps occur:
        # 1. The coroutine encounters the `await` keyword, indicating that it needs to pause its execution and wait for the awaited operation
        #    to complete.
        # 2. In this case, `asyncio.sleep(3)` is an asynchronous function that returns a coroutine that completes after a given time delay. 
        #    It schedules a future event that will be triggered after the specified time.
        # 3. The coroutine invoking `await asyncio.sleep(3)` registers itself with the event loop, indicating that it is waiting for the 
        #    completion of the sleep operation.
        # 4. The event loop takes control and continues executing other ready coroutines or tasks that are not waiting for any resources.
        # 5. During the sleep period, the event loop is free to handle other coroutines, perform I/O operations, or execute other tasks.
        # 6. After the specified time delay (in this case, 3 seconds), the event loop receives a notification that the sleep operation 
        #    has completed.
        # 7. The event loop then resumes the execution of the coroutine that was waiting for the sleep operation, allowing it to proceed 
        #    to the next line of code.

        # In summary, when you encounter `await` in a coroutine, it suspends the current coroutine's execution, allows other coroutines to run,
        # and resumes the execution of the awaiting coroutine when the awaited operation is complete. This cooperative multitasking enables 
        # efficient utilization of system resources in asynchronous programming.
        
        # we can create a coroutine, all we need is at somepoint
        # have a await asyinc.sleep() so we can let other coroutines get executed as well
        # this is the base minimum, there are many more things to learn!
        # for i in range(1_000_000):
        #     for c in itertools.cycle('#*'):
        #         print(f'\r{c} its heavyprocessing...', end='', flush=True)
                # await asyncio.sleep(0.05)
        await asyncio.sleep(0.3)

        return 44
    
    async def load_animation_coroutine(msg:str):
        for ch in itertools.cycle(r'\|/-'):
            label = f'\r{ch} {msg}'        
            print(label, end='', flush=True)
            try:
                # a coroutine yields control explicitly with the await keyword.
                # that is, it gives up control to the event loop, so other coroutines can work!
                # here the current coroutine (our ffunction with async is a coroutine here)
                # is signaling that its going to give up control to sleep for (0.1) seconds
                # and then when sleep is done, the scheduler resumes our coroutine again
                # this is in a loop, so the effect is, each time, it prints sth and sleeps
                # for 0.1 seconds, and this goes on until we break!
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
        print(f"\r{''*len(label)}\r", end='')
        print(f'coroutine animation ended')

    # creates the load_animation_coroutine task to schedule it to run eventually 
    # this is run by the asyncio.run object, basically all coroutines are handled by that
    # and if you print this object youll notice it says pending! 
    # in short, asyncio.create_task, runs a coroutine, the same way a thread runs a callable!
    # so create_task requires a coroutine to work!
    task = asyncio.create_task(load_animation_coroutine('coroutine please wait...'))
    print(f'coroutine: {task}')
    
    # this means calling a coroutine from another coroutine, and it blocks the current coroutine
    # until its job is done, meanwhilte the asyncio driver(event loop) is managing different coroutines up
    # to this point, that is, our task, plus this heavy_process, and anyother ones are all being handled
    # by that driver, it interleaves between them so we can both see the animation coroutine and also
    # heavy_process output at the same time (seemingly, they are not in parallel, but concurrent, interleaved that is)
    # blocks the execution of our coroutine (i.e. run_coroutine_animation) until the heavyprocess is done
    # if we remove await here, we wont get the output of heavy_process, 
    # we get a runtime warning saying the heavy_process was never awaited!
    # if we replace asyncio.sleep with time.sleep, our task never gets executed that is : 
    # time.sleep(3) blocks for 3 seconds; nothing else can happen in the program,
    # because the main thread is blocked—and it is the only thread. The operating sys‐
    # tem will continue with other activities. After 3 seconds, sleep unblocks, and
    # heavy_process returns.
    #
    # in otherwords: 
    # Never use time.sleep(…) in asyncio coroutines unless you want
    # to pause your whole program. If a coroutine needs to spend some
    # time doing nothing, it should await asyncio.sleep(DELAY). This
    # yields control back to the asyncio event loop, which can drive
    # other pending coroutines.
    
    
    result = await heavy_process()
    # starts an asyncio.CancellError inside our task, instructing it to exit
    task.cancel()
    # now that the result is ready return it
    return result

     
    
def main_coroutine():
    # the return value of asyncio.run comes from whatever its calling, i.e. run_coroutine_animation()
    result = asyncio.run(run_coroutine_animation())
    print(f'result: {result}')



    
if __name__ == "__main__":
    # main_thread()
    # main_multiprocessing()
    main_coroutine()