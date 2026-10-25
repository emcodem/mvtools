/*****************************************************************************

        AvstpFallbackPool.cpp

--- Legal stuff ---

This program is free software. It comes without any warranty, to
the extent permitted by applicable law. You can redistribute it
and/or modify it under the terms of the Do What The Fuck You Want
To Public License, Version 2, as published by Sam Hocevar. See
http://sam.zoy.org/wtfpl/COPYING for more details.

*Tab=3***********************************************************************/



/*\\\ INCLUDE FILES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/

#include "AvstpFallbackPool.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>



/*\\\ PRIVATE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/



namespace
{

// One per avstp_TaskDispatcher: counts the tasks of that batch still
// outstanding, so wait_completion() can block until it reaches zero.
struct Dispatcher
{
	std::atomic <int>       _pending { 0 };
	std::mutex              _mtx;
	std::condition_variable _cv;
};

struct Job
{
	avstp_TaskPtr   _task_ptr;
	avstp_TaskDispatcher *
	                _td_ptr;
	void *          _user_data_ptr;
	Dispatcher *    _disp_ptr;
};

class Pool
{
public:

	Pool ()
	{
		const int      n = AvstpFallbackPool::get_nbr_threads ();
		_workers.reserve (n);
		for (int i = 0; i < n; ++i)
		{
			_workers.emplace_back (&Pool::worker_loop, this);
		}
	}

	// Lets the workers finish the queued jobs, then joins them.
	~Pool ()
	{
		{
			std::lock_guard <std::mutex>	lock (_queue_mtx);
			_stop_flag = true;
		}
		_queue_cv.notify_all ();
		for (auto &th : _workers)
		{
			th.join ();
		}
	}

	void	submit (const Job &job)
	{
		{
			std::lock_guard <std::mutex>	lock (_queue_mtx);
			_queue.push (job);
		}
		_queue_cv.notify_one ();
	}

private:

	void	worker_loop ()
	{
		for (;;)
		{
			Job	job;
			{
				std::unique_lock <std::mutex>	lock (_queue_mtx);
				_queue_cv.wait (lock, [this] { return (_stop_flag || ! _queue.empty ()); });
				if (_queue.empty ())
				{
					return;
				}
				job = _queue.front ();
				_queue.pop ();
			}

			job._task_ptr (job._td_ptr, job._user_data_ptr);

			// The decrement must happen under the Dispatcher's mutex. Once
			// wait_completion() can see _pending == 0 it returns and the
			// caller deletes the Dispatcher, so this thread must not touch it
			// after that point. Decrementing outside the lock (and locking only
			// to notify) let the waiter see 0 first - on entry or after a
			// spurious wakeup - and free the Dispatcher while this thread was
			// about to lock its mutex: a use-after-free that corrupted the
			// next Dispatcher allocated at the same address and deadlocked.
			{
				std::lock_guard <std::mutex>	lock (job._disp_ptr->_mtx);
				if (job._disp_ptr->_pending.fetch_sub (1, std::memory_order_acq_rel) == 1)
				{
					job._disp_ptr->_cv.notify_all ();
				}
			}
		}
	}

	std::vector <std::thread>	_workers;
	std::mutex                 _queue_mtx;
	std::condition_variable    _queue_cv;
	std::queue <Job>           _queue;
	bool                       _stop_flag = false;

	               Pool (const Pool &other)            = delete;
	               Pool (Pool &&other)                 = delete;
	Pool &         operator = (const Pool &other)       = delete;
	Pool &         operator = (Pool &&other)            = delete;

};	// class Pool

// Deliberately a raw pointer: it is deleted by shutdown(), never by a
// static destructor, which would join under the loader lock.
std::mutex	g_pool_mtx;
Pool *		g_pool_ptr = 0;

}	// namespace



/*\\\ PUBLIC \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/



int	AvstpFallbackPool::get_nbr_threads ()
{
	static const int	nbr_threads =
		std::max (1, int (std::thread::hardware_concurrency ()));

	return (nbr_threads);
}



avstp_TaskDispatcher *	AvstpFallbackPool::create_dispatcher ()
{
	return (reinterpret_cast <avstp_TaskDispatcher *> (new Dispatcher));
}



void	AvstpFallbackPool::destroy_dispatcher (avstp_TaskDispatcher *td_ptr)
{
	delete reinterpret_cast <Dispatcher *> (td_ptr);
}



int	AvstpFallbackPool::enqueue_task (avstp_TaskDispatcher *td_ptr, avstp_TaskPtr task_ptr, void *user_data_ptr)
{
	int            ret_val = avstp_Err_OK;

	if (td_ptr == 0 || task_ptr == 0)
	{
		ret_val = avstp_Err_INVALID_ARG;
	}
	else
	{
		Dispatcher *	disp_ptr = reinterpret_cast <Dispatcher *> (td_ptr);

		disp_ptr->_pending.fetch_add (1, std::memory_order_relaxed);

		Job	job;
		job._task_ptr      = task_ptr;
		job._td_ptr        = td_ptr;
		job._user_data_ptr = user_data_ptr;
		job._disp_ptr      = disp_ptr;

		std::lock_guard <std::mutex>	lock (g_pool_mtx);
		if (g_pool_ptr == 0)
		{
			g_pool_ptr = new Pool;
		}
		g_pool_ptr->submit (job);
	}

	return (ret_val);
}



int	AvstpFallbackPool::wait_completion (avstp_TaskDispatcher *td_ptr)
{
	int            ret_val = avstp_Err_OK;

	if (td_ptr == 0)
	{
		ret_val = avstp_Err_INVALID_ARG;
	}
	else
	{
		Dispatcher *	disp_ptr = reinterpret_cast <Dispatcher *> (td_ptr);

		std::unique_lock <std::mutex>	lock (disp_ptr->_mtx);
		disp_ptr->_cv.wait (lock, [disp_ptr]
		{
			return (disp_ptr->_pending.load (std::memory_order_acquire) == 0);
		});
	}

	return (ret_val);
}



void	AvstpFallbackPool::shutdown ()
{
	Pool *		pool_ptr = 0;
	{
		std::lock_guard <std::mutex>	lock (g_pool_mtx);
		pool_ptr   = g_pool_ptr;
		g_pool_ptr = 0;
	}
	delete pool_ptr;
}



/*\\\ EOF \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*/
